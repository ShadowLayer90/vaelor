import { Button } from "./ui";

/**
 * "Page 1 of 2 · 12 items" with Previous and Next, as the History and Memory
 * boards draw it. The paging itself is the shared `usePagination` (client-side,
 * over the list the page already holds); only the bar's layout is the boards':
 * History puts the words on the left and both buttons on the right, Memory
 * puts Previous and Next at either end with the words between them.
 *
 * One page draws no bar, as the shared PaginationControls never did.
 */
export function ListPager({
  label,
  layout = "end",
  page,
  setPage,
  totalItems,
  totalPages,
}: {
  label: string;
  layout?: "end" | "ends";
  page: number;
  setPage: (next: number | ((current: number) => number)) => void;
  totalItems: number;
  totalPages: number;
}) {
  if (totalPages <= 1) return null;
  const words = <span className="as-small as-muted">Page {page} of {totalPages} · {totalItems} items</span>;
  const previous = (
    <Button disabled={page === 1} onClick={() => setPage((current) => Math.max(1, current - 1))} type="button">Previous</Button>
  );
  const next = (
    <Button disabled={page === totalPages} onClick={() => setPage((current) => Math.min(totalPages, current + 1))} type="button">Next</Button>
  );
  return (
    <nav aria-label={`${label} pages`} className="as-pager">
      {layout === "ends"
        ? <>{previous}{words}{next}</>
        : <>{words}<span className="as-pager__buttons">{previous}{next}</span></>}
    </nav>
  );
}
