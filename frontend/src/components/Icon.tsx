import type { SVGProps } from "react";

export type IconName =
  | "activity"
  | "alert"
  | "atx"
  | "bolt"
  | "chevron"
  | "cpu"
  | "database"
  | "display"
  | "download"
  | "fan"
  | "gpio"
  | "gpu"
  | "grid"
  | "hdmi"
  | "lock"
  | "logout"
  | "memory"
  | "network"
  | "npu"
  | "nvme"
  | "oled"
  | "power"
  | "settings"
  | "shield"
  | "terminal"
  | "upload"
  | "usb";

const paths: Record<IconName, React.ReactNode> = {
  activity: <path d="M3 12h4l2.3-7 4.2 14 2.2-7H21" />,
  alert: (
    <>
      <path d="M12 3 2.8 19h18.4L12 3Z" />
      <path d="M12 9v4M12 16.5v.5" />
    </>
  ),
  atx: (
    <>
      <rect x="4" y="5" width="16" height="14" rx="2" />
      <path d="M8 9h2m4 0h2M8 13h2m4 0h2M8 17h2m4 0h2" />
    </>
  ),
  bolt: <path d="m13 2-8 12h7l-1 8 8-12h-7l1-8Z" />,
  chevron: <path d="m9 18 6-6-6-6" />,
  cpu: (
    <>
      <rect x="7" y="7" width="10" height="10" rx="2" />
      <path d="M9 2v3m6-3v3M9 19v3m6-3v3M2 9h3m-3 6h3m14-6h3m-3 6h3M10 10h4v4h-4z" />
    </>
  ),
  database: (
    <>
      <ellipse cx="12" cy="5" rx="8" ry="3" />
      <path d="M4 5v7c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12v7c0 1.7 3.6 3 8 3s8-1.3 8-3v-7" />
    </>
  ),
  display: (
    <>
      <rect x="3" y="4" width="18" height="13" rx="2" />
      <path d="M8 21h8m-4-4v4" />
    </>
  ),
  download: <path d="M12 3v12m-5-5 5 5 5-5M4 20h16" />,
  fan: (
    <>
      <circle cx="12" cy="12" r="2" />
      <path d="M12 10c-1-3-1-6 1-7 2-1 4 1 3 3-.7 2-2.2 3.3-4 4ZM14 12c3-1 6-1 7 1 1 2-1 4-3 3-2-.7-3.3-2.2-4-4ZM12 14c1 3 1 6-1 7-2 1-4-1-3-3 .7-2 2.2-3.3 4-4ZM10 12c-3 1-6 1-7-1-1-2 1-4 3-3 2 .7 3.3 2.2 4 4Z" />
    </>
  ),
  gpio: (
    <>
      <path d="M5 5v14M19 5v14" />
      <circle cx="8.5" cy="7" r="1" />
      <circle cx="8.5" cy="12" r="1" />
      <circle cx="8.5" cy="17" r="1" />
      <circle cx="15.5" cy="7" r="1" />
      <circle cx="15.5" cy="12" r="1" />
      <circle cx="15.5" cy="17" r="1" />
    </>
  ),
  // A discrete graphics board: a card edge, its die, and the fins over it.
  gpu: (
    <>
      <path d="M3 6h16a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H7l-4 3V6Z" />
      <path d="M8 10h6v5H8z" />
      <path d="M16.5 10v5M18.5 10v5" />
    </>
  ),
  grid: <path d="M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z" />,
  /*
   * The third compute engine. There was a `gpu` glyph and no `npu` one, so the
   * neural accelerator had no way to appear beside the parts it sits next to.
   * A die with two ranks of nodes and the links between them: recognisably not
   * the graphics card, and recognisably not a plain processor.
   */
  npu: (
    <>
      <path d="M8 8h8v8H8z" />
      <path d="M12 3v5M12 16v5M3 12h5M16 12h5" />
      <path d="M8.5 3v5M15.5 3v5M8.5 16v5M15.5 16v5" />
    </>
  ),
  hdmi: (
    <>
      <path d="M5 6h14l2 4-2 8H5l-2-8 2-4Z" />
      <path d="M7 10h10M8 14h8" />
    </>
  ),
  lock: (
    <>
      <rect x="5" y="10" width="14" height="11" rx="2" />
      <path d="M8 10V7a4 4 0 0 1 8 0v3" />
    </>
  ),
  logout: <path d="M10 5H5v14h5m4-3 4-4-4-4m4 4H9" />,
  memory: (
    <>
      <rect x="4" y="7" width="16" height="10" rx="2" />
      <path d="M8 10v4m4-4v4m4-4v4M7 4v3m5-3v3m5-3v3M7 17v3m5-3v3m5-3v3" />
    </>
  ),
  network: (
    <>
      <path d="M5 12.6a10 10 0 0 1 14 0M8.5 16a5 5 0 0 1 7 0" />
      <path d="M12 20h.01M2 9a15 15 0 0 1 20 0" />
    </>
  ),
  nvme: (
    <>
      <rect x="3" y="7" width="18" height="10" rx="2" />
      <path d="M6 10h5v4H6zM15 10h3m-3 3h3M7 17v2m3-2v2m4-2v2m3-2v2" />
    </>
  ),
  oled: (
    <>
      <rect x="3" y="5" width="18" height="14" rx="2" />
      <path d="M6 15h3l2-6 2.5 7 1.5-4h3" />
    </>
  ),
  power: <path d="M12 2v10m6.4-6.4a9 9 0 1 1-12.8 0" />,
  settings: (
    <>
      <circle cx="12" cy="12" r="3" />
      <path d="M19 12a7 7 0 0 0-.1-1l2-1.5-2-3.4-2.4 1A7 7 0 0 0 15 6l-.3-2.6h-4L10.4 6A7 7 0 0 0 8 7.1l-2.4-1-2 3.4 2 1.5a7 7 0 0 0 0 2l-2 1.5 2 3.4 2.4-1A7 7 0 0 0 10.4 18l.3 2.6h4L15 18a7 7 0 0 0 2-1.1l2.4 1 2-3.4-2-1.5a7 7 0 0 0 .1-1Z" />
    </>
  ),
  shield: <path d="M12 3 5 6v5c0 4.8 2.8 8.2 7 10 4.2-1.8 7-5.2 7-10V6l-7-3Zm-3 9 2 2 4-5" />,
  terminal: <path d="m4 7 5 5-5 5m8 0h8" />,
  upload: <path d="M12 16V4m-5 5 5-5 5 5M4 20h16" />,
  usb: (
    <>
      <path d="M12 3v13m0 0-4-4m4 4 4-4M8 12V8" />
      <circle cx="8" cy="7" r="1.3" />
      <path d="M16 12V8h2" />
      <rect x="17.3" y="6.7" width="2.6" height="2.6" rx=".3" />
      <path d="M12 16v3" />
      <circle cx="12" cy="20" r="1.5" />
    </>
  ),
};

export function Icon({
  name,
  size = 20,
  ...props
}: { name: IconName; size?: number } & SVGProps<SVGSVGElement>) {
  return (
    <svg
      aria-hidden="true"
      fill="none"
      height={size}
      viewBox="0 0 24 24"
      width={size}
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="1.7"
      {...props}
    >
      {paths[name]}
    </svg>
  );
}
