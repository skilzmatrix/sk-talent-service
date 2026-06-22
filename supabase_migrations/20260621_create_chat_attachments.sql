-- Chat attachment metadata for storage-backed uploads
create extension if not exists pgcrypto;

create table if not exists public.chat_attachments (
    id uuid primary key default gen_random_uuid(),
    conversation_id uuid null,
    file_name text not null,
    mime_type text not null,
    size_bytes bigint not null check (size_bytes >= 0),
    storage_key text not null unique,
    bucket text not null default 'candidate_resumes',
    created_at timestamptz not null default now()
);

create index if not exists idx_chat_attachments_conversation_id
    on public.chat_attachments (conversation_id);

create index if not exists idx_chat_attachments_created_at
    on public.chat_attachments (created_at desc);
