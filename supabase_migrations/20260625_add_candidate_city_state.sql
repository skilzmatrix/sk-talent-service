alter table if exists public.candidates
  add column if not exists city text,
  add column if not exists state text;

create index if not exists idx_candidates_city on public.candidates(city);
create index if not exists idx_candidates_state on public.candidates(state);
