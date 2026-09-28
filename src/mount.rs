//! A read-only FUSE filesystem over a [`Lazy`] model: the model's files appear at once,
//! and reads fetch what they need. Loaders that `mmap` safetensors work unchanged, since
//! page faults become reads.

use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, UNIX_EPOCH};

use anyhow::Result;
use fuser::{
    Errno, FileAttr, FileHandle, FileType, Filesystem, Generation, INodeNo, LockOwner, MountOption,
    OpenFlags, ReplyAttr, ReplyData, ReplyDirectory, ReplyEntry, Request,
};

use crate::lazy::Lazy;

/// The files never change, so the kernel may cache names and attributes for long.
const TTL: Duration = Duration::from_secs(3600);

enum Node {
    Dir(BTreeMap<String, u64>),
    /// Index into the manifest's files.
    File(usize),
}

struct ModelFs {
    lazy: Arc<Lazy>,
    runtime: tokio::runtime::Handle,
    /// Inode `i + 1` is `nodes[i]`; inode 1 is the root directory.
    nodes: Vec<Node>,
    parents: Vec<u64>,
}

impl ModelFs {
    fn new(lazy: Arc<Lazy>, runtime: tokio::runtime::Handle) -> ModelFs {
        let mut nodes = vec![Node::Dir(BTreeMap::new())];
        let mut parents = vec![1];
        for (i, f) in lazy.manifest().files.iter().enumerate() {
            let parts: Vec<&str> = f.path.split('/').collect();
            let mut dir = 1u64;
            for (depth, part) in parts.iter().enumerate() {
                let leaf = depth + 1 == parts.len();
                let existing = match &nodes[dir as usize - 1] {
                    Node::Dir(children) => children.get(*part).copied(),
                    Node::File(_) => None,
                };
                dir = match existing {
                    Some(ino) => ino,
                    None => {
                        nodes.push(if leaf {
                            Node::File(i)
                        } else {
                            Node::Dir(BTreeMap::new())
                        });
                        parents.push(dir);
                        let ino = nodes.len() as u64;
                        if let Node::Dir(children) = &mut nodes[dir as usize - 1] {
                            children.insert(part.to_string(), ino);
                        }
                        ino
                    }
                };
            }
        }
        ModelFs {
            lazy,
            runtime,
            nodes,
            parents,
        }
    }

    fn node(&self, ino: INodeNo) -> Option<&Node> {
        self.nodes.get(u64::from(ino).checked_sub(1)? as usize)
    }

    fn attr(&self, ino: u64) -> Option<FileAttr> {
        let (kind, size, perm, nlink) = match self.nodes.get(ino as usize - 1)? {
            Node::Dir(_) => (FileType::Directory, 0, 0o555, 2),
            Node::File(i) => (
                FileType::RegularFile,
                self.lazy.manifest().files[*i].size,
                0o444,
                1,
            ),
        };
        Some(FileAttr {
            ino: INodeNo(ino),
            size,
            blocks: size.div_ceil(512),
            atime: UNIX_EPOCH,
            mtime: UNIX_EPOCH,
            ctime: UNIX_EPOCH,
            crtime: UNIX_EPOCH,
            kind,
            perm,
            nlink,
            uid: owner_ids().0,
            gid: owner_ids().1,
            rdev: 0,
            flags: 0,
            blksize: 1 << 16,
        })
    }
}

/// The mounting user's ids, so files show as theirs.
fn owner_ids() -> (u32, u32) {
    // SAFETY: getuid and getgid can't fail and have no preconditions.
    unsafe { (libc::getuid(), libc::getgid()) }
}

impl Filesystem for ModelFs {
    fn lookup(&self, _req: &Request, parent: INodeNo, name: &OsStr, reply: ReplyEntry) {
        let found = match (self.node(parent), name.to_str()) {
            (Some(Node::Dir(children)), Some(name)) => children.get(name).copied(),
            _ => None,
        };
        match found.and_then(|ino| self.attr(ino)) {
            Some(attr) => reply.entry(&TTL, &attr, Generation(0)),
            None => reply.error(Errno::ENOENT),
        }
    }

    fn getattr(&self, _req: &Request, ino: INodeNo, _fh: Option<FileHandle>, reply: ReplyAttr) {
        match self.attr(ino.into()) {
            Some(attr) => reply.attr(&TTL, &attr),
            None => reply.error(Errno::ENOENT),
        }
    }

    fn read(
        &self,
        _req: &Request,
        ino: INodeNo,
        _fh: FileHandle,
        offset: u64,
        size: u32,
        _flags: OpenFlags,
        _lock_owner: Option<LockOwner>,
        reply: ReplyData,
    ) {
        let Some(Node::File(file)) = self.node(ino) else {
            reply.error(Errno::EISDIR);
            return;
        };
        // Answer from the runtime, so one read waiting on the network doesn't hold up
        // the others.
        let (lazy, file) = (self.lazy.clone(), *file);
        self.runtime.spawn(async move {
            match lazy.read(file, offset, size as usize).await {
                Ok(data) => reply.data(&data),
                Err(e) => {
                    eprintln!("read failed: {e:#}");
                    reply.error(Errno::EIO);
                }
            }
        });
    }

    fn readdir(
        &self,
        _req: &Request,
        ino: INodeNo,
        _fh: FileHandle,
        offset: u64,
        mut reply: ReplyDirectory,
    ) {
        let Some(Node::Dir(children)) = self.node(ino) else {
            reply.error(Errno::ENOTDIR);
            return;
        };
        let parent = self.parents[u64::from(ino) as usize - 1];
        let entries = [
            (u64::from(ino), FileType::Directory, "."),
            (parent, FileType::Directory, ".."),
        ]
        .into_iter()
        .chain(children.iter().map(|(name, &child)| {
            let kind = match self.nodes[child as usize - 1] {
                Node::Dir(_) => FileType::Directory,
                Node::File(_) => FileType::RegularFile,
            };
            (child, kind, name.as_str())
        }));
        for (i, (child, kind, name)) in entries.enumerate().skip(offset as usize) {
            if reply.add(INodeNo(child), (i + 1) as u64, kind, name) {
                break;
            }
        }
        reply.ok();
    }
}

/// A mounted model. Dropping it unmounts.
pub struct Mounted {
    session: Option<fuser::BackgroundSession>,
}

impl Mounted {
    pub fn unmount(mut self) -> Result<()> {
        if let Some(s) = self.session.take() {
            s.umount_and_join()?;
        }
        Ok(())
    }
}

impl Drop for Mounted {
    fn drop(&mut self) {
        if let Some(s) = self.session.take() {
            let _ = s.umount_and_join();
        }
    }
}

/// Mount `lazy` read-only at `dir`, served by threads outside the tokio runtime; reads
/// run on `runtime`.
pub fn mount(lazy: Arc<Lazy>, dir: &Path, runtime: tokio::runtime::Handle) -> Result<Mounted> {
    let fs = ModelFs::new(lazy, runtime);
    let mut config = fuser::Config::default();
    config.mount_options = vec![
        MountOption::RO,
        MountOption::FSName("chungus".into()),
        MountOption::Subtype("chungus".into()),
    ];
    config.n_threads = Some(4);
    let session = fuser::spawn_mount(fs, dir, &config)
        .map_err(|e| anyhow::anyhow!("mount {}: {e} (is FUSE installed?)", dir.display()))?;
    Ok(Mounted {
        session: Some(session),
    })
}
