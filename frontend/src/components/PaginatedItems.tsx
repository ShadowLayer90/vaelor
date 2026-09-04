import { Button } from "./ui";
import { useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

export function usePagination<T>(items: T[], pageSize: number) {
  const [page, setPage] = useState(1);
  const totalPages = Math.max(1, Math.ceil(items.length / pageSize));
  useEffect(() => {
    setPage((current) => Math.min(current, totalPages));
  }, [totalPages]);
  const visible = useMemo(
    () => items.slice((page - 1) * pageSize, page * pageSize),
    [items, page, pageSize],
  );
  return { page, setPage, totalPages, visible };
}

export function PaginationControls({
  page,
  setPage,
  totalPages,
  totalItems,
  label,
}: {
  page: number;
  setPage: (next: number | ((current: number) => number)) => void;
  totalPages: number;
  totalItems: number;
  label: string;
}) {
  if (totalPages <= 1) return null;
  return (
    <nav aria-label={`${label} pages`} className="pagination">
      <Button variant="quiet"

        disabled={page === 1}
        onClick={() => setPage((current) => Math.max(1, current - 1))}
        type="button"
      >
        Previous
      </Button>
      <span>
        Page <strong>{page}</strong> of <strong>{totalPages}</strong>
        <small>{totalItems} items</small>
      </span>
      <Button variant="quiet"

        disabled={page === totalPages}
        onClick={() => setPage((current) => Math.min(totalPages, current + 1))}
        type="button"
      >
        Next
      </Button>
    </nav>
  );
}

export function PaginatedItems<T>({
  items,
  pageSize,
  label,
  render,
}: {
  items: T[];
  pageSize: number;
  label: string;
  render: (item: T, index: number) => ReactNode;
}) {
  const { page, setPage, totalPages, visible } = usePagination(items, pageSize);

  return (
    <>
      {visible.map((item, index) => render(item, (page - 1) * pageSize + index))}
      <PaginationControls
        label={label}
        page={page}
        setPage={setPage}
        totalItems={items.length}
        totalPages={totalPages}
      />
    </>
  );
}
