"use client";

import { useCallback, useEffect, useRef, useState } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { FileCode, Download, Activity, Trash2, Upload, X, CheckCircle } from 'lucide-react';
import api, { downloadFile } from '@/lib/api';

const ALLOWED_EXT = ['.csv', '.txt', '.json', '.ppg', '.edf', '.xml', '.pdf', '.xlsx', '.xls'];

interface FileItem { name: string; owner: string; size: number; type: string; created_at: string; }
interface UploadState { file: File; progress: 'idle' | 'uploading' | 'done' | 'error'; error?: string; }

function fmtSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function Files() {
  const router = useRouter();
  const [files, setFiles] = useState<FileItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploads, setUploads] = useState<UploadState[]>([]);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const fetchFiles = useCallback(async () => {
    try {
      const res = await api.get('/files/local/');
      if (res.data.is_superuser) {
        const all: FileItem[] = [];
        res.data.users.forEach((u: any) => u.files.forEach((f: any) => all.push({ ...f, owner: u.username })));
        setFiles(all);
      } else {
        setFiles(res.data.files.map((f: any) => ({ ...f, owner: res.data.username })));
      }
    } catch (err: any) {
      if (err.response?.status === 401) router.push('/login');
    } finally { setLoading(false); }
  }, [router]);

  useEffect(() => { fetchFiles(); }, [fetchFiles]);

  const deleteFile = async (owner: string, filename: string) => {
    if (!confirm('Delete this file?')) return;
    try { await api.delete(`/files/local/${owner}/${filename}/`); fetchFiles(); }
    catch { alert('Failed to delete file.'); }
  };

  const downloadPdf = async (filename: string) => {
    try { await downloadFile(`/hl7/pdf/${filename}/`, `${filename}.pdf`); }
    catch { alert('Failed to download HL7 PDF.'); }
  };

  const uploadFiles = async (fileList: File[]) => {
    const valid = fileList.filter((f) => {
      const ext = '.' + f.name.split('.').pop()?.toLowerCase();
      return ALLOWED_EXT.includes(ext);
    });
    if (valid.length !== fileList.length) {
      alert(`Some files were skipped. Allowed types: ${ALLOWED_EXT.join(', ')}`);
    }
    if (!valid.length) return;

    const newUploads: UploadState[] = valid.map((f) => ({ file: f, progress: 'idle' }));
    setUploads((p) => [...newUploads, ...p]);

    for (let i = 0; i < valid.length; i++) {
      setUploads((p) => p.map((u, idx) => idx === i ? { ...u, progress: 'uploading' } : u));
      try {
        const fd = new FormData();
        fd.append('file', valid[i]);
        await api.post('/files/local/upload/', fd, { headers: { 'Content-Type': 'multipart/form-data' } });
        setUploads((p) => p.map((u, idx) => idx === i ? { ...u, progress: 'done' } : u));
      } catch (err: any) {
        const msg = err.response?.data?.error || 'Upload failed.';
        setUploads((p) => p.map((u, idx) => idx === i ? { ...u, progress: 'error', error: msg } : u));
      }
    }
    fetchFiles();
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault(); setDragging(false);
    uploadFiles(Array.from(e.dataTransfer.files));
  };
  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) uploadFiles(Array.from(e.target.files));
    e.target.value = '';
  };

  if (loading) return <div className="container" style={{ padding: '4rem', textAlign: 'center' }}>Loading files…</div>;

  return (
    <div className="container animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
        <h2>My Files</h2>
        <button onClick={() => inputRef.current?.click()} className="btn btn-primary" style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <Upload size={17} /> Upload files
        </button>
        <input ref={inputRef} type="file" multiple accept={ALLOWED_EXT.join(',')} style={{ display: 'none' }} onChange={onInputChange} />
      </div>

      {/* Drop zone */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
        style={{
          border: `2px dashed ${dragging ? 'var(--primary)' : 'var(--border)'}`,
          borderRadius: 'var(--radius-md)', padding: '2rem', textAlign: 'center', cursor: 'pointer',
          background: dragging ? 'rgba(16,185,129,0.05)' : 'transparent',
          transition: 'all 0.15s', color: 'var(--text-muted)',
        }}>
        <Upload size={28} style={{ marginBottom: '0.5rem', opacity: 0.4 }} />
        <p style={{ fontWeight: 600, marginBottom: '0.25rem' }}>Drag & drop files here</p>
        <p style={{ fontSize: '0.8rem' }}>or click to browse · {ALLOWED_EXT.join(' ')}</p>
      </div>

      {/* Upload progress */}
      {uploads.length > 0 && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          {uploads.map((u, i) => (
            <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', padding: '0.7rem 1rem', background: 'var(--info-bg)', border: '1px solid var(--info-border)', borderRadius: 'var(--radius-sm)', fontSize: '0.85rem' }}>
              {u.progress === 'done' ? <CheckCircle size={16} color="var(--success)" /> : u.progress === 'error' ? <X size={16} color="var(--error)" /> : <div style={{ width: 16, height: 16, border: '2px solid var(--primary)', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 0.7s linear infinite' }} />}
              <span style={{ flex: 1 }}>{u.file.name}</span>
              {u.error && <span style={{ color: 'var(--error)', fontSize: '0.8rem' }}>{u.error}</span>}
              {u.progress === 'done' && <span style={{ color: 'var(--success)', fontSize: '0.8rem' }}>Uploaded</span>}
              <button onClick={() => setUploads((p) => p.filter((_, j) => j !== i))} style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text-muted)', padding: 0, display: 'flex' }}><X size={14} /></button>
            </div>
          ))}
        </div>
      )}

      {/* File table */}
      <div className="glass-panel" style={{ overflow: 'hidden' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
          <thead>
            <tr style={{ background: 'var(--panel-bg-light)', borderBottom: '1px solid var(--border)' }}>
              <th style={{ padding: '0.9rem 1.25rem' }}>File Name</th>
              <th style={{ padding: '0.9rem 1rem' }}>Owner</th>
              <th style={{ padding: '0.9rem 1rem' }}>Type</th>
              <th style={{ padding: '0.9rem 1rem' }}>Size</th>
              <th style={{ padding: '0.9rem 1.25rem', textAlign: 'right' }}>Actions</th>
            </tr>
          </thead>
          <tbody>
            {files.length === 0 ? (
              <tr>
                <td colSpan={5} style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
                  No files yet. Upload some above.
                </td>
              </tr>
            ) : (
              files.map((f, i) => (
                <tr key={i} style={{ borderBottom: '1px solid var(--border)' }}>
                  <td style={{ padding: '0.9rem 1.25rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <FileCode size={17} color="var(--primary)" />
                      <span style={{ fontWeight: 500 }}>{f.name}</span>
                    </div>
                  </td>
                  <td style={{ padding: '0.9rem 1rem', color: 'var(--text-muted)', fontSize: '0.88rem' }}>{f.owner}</td>
                  <td style={{ padding: '0.9rem 1rem', color: 'var(--text-muted)', fontSize: '0.88rem' }}>{f.type?.split('/').pop() || '—'}</td>
                  <td style={{ padding: '0.9rem 1rem', color: 'var(--text-muted)', fontSize: '0.88rem' }}>{fmtSize(f.size)}</td>
                  <td style={{ padding: '0.9rem 1.25rem' }}>
                    <div style={{ display: 'flex', gap: '0.4rem', justifyContent: 'flex-end' }}>
                      {f.name.toLowerCase().endsWith('.csv') && (
                        <Link href={`/visualization/ppg/${f.owner}/${f.name}`} className="btn btn-outline" style={{ padding: '0.45rem 0.6rem' }} title="Visualize">
                          <Activity size={15} color="var(--secondary)" />
                        </Link>
                      )}
                      <button onClick={() => downloadPdf(f.name)} className="btn btn-outline" style={{ padding: '0.45rem 0.6rem' }} title="HL7 PDF">
                        <Download size={15} color="var(--success)" />
                      </button>
                      <button onClick={() => deleteFile(f.owner, f.name)} className="btn btn-outline" style={{ padding: '0.45rem 0.6rem', borderColor: 'var(--error)', color: 'var(--error)' }} title="Delete">
                        <Trash2 size={15} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
