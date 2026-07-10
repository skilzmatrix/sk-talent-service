alter table if exists public.resumes
  add column if not exists experience text;

alter table if exists public.candidates
  add column if not exists experience text;

create index if not exists idx_candidates_experience on public.candidates(experience);
