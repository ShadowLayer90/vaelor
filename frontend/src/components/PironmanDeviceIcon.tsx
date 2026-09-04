import { useId } from "react";

interface Props {
  model: string;
  size?: number;
  label?: string;
}

type ShellProps = {
  children: (defsId: string) => React.ReactNode;
  height?: number;
  label?: string;
  model: string;
  size: number;
};

function TechnicalShell({ children, height = 220, label, model, size }: ShellProps) {
  const titleId = useId();
  const defsId = `${titleId}-enclosure`;
  return (
    <svg
      aria-hidden={label ? undefined : "true"}
      aria-labelledby={label ? titleId : undefined}
      className={`pironman-device-icon pironman-device-icon--${model}`}
      data-enclosure={model}
      height={size}
      role={label ? "img" : undefined}
      viewBox={`0 0 240 ${height}`}
      width={size}
    >
      {label && <title id={titleId}>{label}</title>}
      <defs>
        <linearGradient id={`${defsId}-panel`} x1="0" x2="1" y1="0" y2="1">
          <stop offset="0" stopColor="#17222b" stopOpacity=".92" />
          <stop offset=".52" stopColor="#070b0f" stopOpacity=".78" />
          <stop offset="1" stopColor="#020304" stopOpacity=".96" />
        </linearGradient>
        <linearGradient id={`${defsId}-metal`} x1="0" x2="1">
          <stop offset="0" stopColor="#d8e0e4" />
          <stop offset=".32" stopColor="#66737c" />
          <stop offset="1" stopColor="#1b2329" />
        </linearGradient>
        <radialGradient id={`${defsId}-fan`}>
          <stop offset="0" stopColor="#09151c" />
          <stop offset=".62" stopColor="#081017" />
          <stop offset=".74" stopColor="#087ca5" />
          <stop offset=".86" stopColor="#00b9ed" />
          <stop offset="1" stopColor="#09131a" />
        </radialGradient>
        <filter id={`${defsId}-glow`} x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur stdDeviation="3" />
        </filter>
      </defs>
      <ellipse cx="120" cy={height - 13} rx="83" ry="9" fill="#000" opacity=".58" />
      {children(defsId)}
    </svg>
  );
}

function Fan({ cx, cy, defsId, radius = 25 }: { cx: number; cy: number; defsId: string; radius?: number }) {
  return (
    <g>
      <circle cx={cx} cy={cy} r={radius + 5} fill="#00aee8" filter={`url(#${defsId}-glow)`} opacity=".14" />
      <circle cx={cx} cy={cy} r={radius} fill={`url(#${defsId}-fan)`} stroke="#29c9f2" strokeWidth="1.4" />
      <circle cx={cx} cy={cy} r={radius - 7} fill="#070b0e" stroke="#ff6a00" strokeOpacity=".52" />
      <g fill="#1db9e8" opacity=".68">
        <path d={`M${cx} ${cy - 4}c-3-14 7-15 12-9 4 6 0 12-7 15Z`} />
        <path d={`M${cx + 4} ${cy}c14-3 15 7 9 12-6 4-12 0-15-7Z`} />
        <path d={`M${cx} ${cy + 4}c3 14-7 15-12 9-4-6 0-12 7-15Z`} />
        <path d={`M${cx - 4} ${cy}c-14 3-15-7-9-12 6-4 12 0 15 7Z`} />
      </g>
      <circle cx={cx} cy={cy} r="4" fill="#ff6a00" />
    </g>
  );
}

function Oled({ x, y }: { x: number; y: number }) {
  return (
    <g>
      <rect x={x} y={y} width="43" height="22" rx="2" fill="#031a20" stroke="#29d7ef" />
      <path d={`M${x + 5} ${y + 7}h12m-12 5h19m-19 5h15`} stroke="#66f0fa" strokeWidth="1.5" />
      <circle cx={x + 36} cy={y + 8} r="2" fill="#40e885" />
      <path d={`M${x + 31} ${y + 16}h7`} stroke="#ff6a00" />
    </g>
  );
}

function Fasteners({ points }: { points: Array<[number, number]> }) {
  return (
    <>
      {points.map(([x, y]) => (
        <g key={`${x}-${y}`}>
          <circle cx={x} cy={y} r="3.2" fill="#050608" stroke="#a7b2b9" strokeWidth=".8" />
          <path d={`M${x - 1.5} ${y}h3`} stroke="#7c8991" strokeWidth=".7" />
        </g>
      ))}
    </>
  );
}

function Standard({ label, size }: Pick<Props, "label" | "size">) {
  return (
    <TechnicalShell label={label} model="pironman5" size={size ?? 92}>
      {(defsId) => <>
      <path d="M52 33 164 23l26 18v151L52 199Z" fill={`url(#${defsId}-panel)`} stroke="#a8b3ba" strokeWidth="2.2" />
      <path d="m164 23 26 18v151l-26-8Z" fill="#05080a" stroke="#44525a" />
      <path d="M52 33 164 23l26 18-111 12Z" fill={`url(#${defsId}-metal)`} opacity=".8" />
      <path d="M67 48h83v128H67z" fill="#071015" fillOpacity=".64" stroke="#b8c1c6" strokeOpacity=".55" />
      <Oled x={75} y={56} />
      <path d="M83 92h42l13 11v44H83z" fill="#0e1519" stroke="#ff6a00" strokeOpacity=".7" />
      <path d="M90 103h39m-39 10h39m-39 10h39" stroke="#57646c" />
      <Fan cx={164} cy={82} defsId={defsId} radius={18} />
      <Fan cx={164} cy={135} defsId={defsId} radius={18} />
      <path d="M78 158h58v8H78z" fill="#13212a" stroke="#00aee8" />
      <circle cx="62" cy="171" r="8" fill="#050607" stroke="#ff6a00" strokeWidth="1.5" />
      <Fasteners points={[[58, 39], [157, 31], [58, 191], [157, 177]]} />
      </>}
    </TechnicalShell>
  );
}

function Max({ label, size }: Pick<Props, "label" | "size">) {
  return (
    <TechnicalShell label={label} model="pironman5-max" size={size ?? 92}>
      {(defsId) => <>
      <path d="m34 46 143-19 29 24v137L34 200Z" fill={`url(#${defsId}-panel)`} stroke="#4b5962" strokeWidth="2.2" />
      <path d="m177 27 29 24v137l-29-9Z" fill="#020406" stroke="#29343b" />
      <path d="M49 58h112v113H49z" fill="#071017" fillOpacity=".75" stroke="#65737c" />
      <Oled x={58} y={64} />
      <path d="M58 102h64v14H58zm0 24h64v14H58z" fill="#0a2433" stroke="#00aee8" />
      <path d="M64 107h42m-42 24h42" stroke="#ff6a00" strokeWidth="2" />
      <path d="M130 60v105" stroke="#ff6a00" strokeOpacity=".45" strokeDasharray="3 5" />
      <Fan cx={171} cy={84} defsId={defsId} radius={24} />
      <Fan cx={171} cy={143} defsId={defsId} radius={24} />
      <circle cx="49" cy="174" r="10" fill="#050607" stroke="#ff6a00" strokeWidth="1.5" />
      <path d="M38 194h143" stroke="#ff6a00" strokeWidth="2" />
      <Fasteners points={[[41, 52], [170, 35], [41, 192], [170, 173]]} />
      </>}
    </TechnicalShell>
  );
}

function Mini({ label, size }: Pick<Props, "label" | "size">) {
  return (
    <TechnicalShell label={label} model="pironman5-mini" size={size ?? 92}>
      {(defsId) => <>
      <path d="m62 58 106-15 23 19v114L62 190Z" fill={`url(#${defsId}-panel)`} stroke="#aab5bb" strokeWidth="2.2" />
      <path d="m168 43 23 19v114l-23-8Z" fill="#030608" stroke="#36434a" />
      <path d="m62 58 106-15 23 19-106 16Z" fill={`url(#${defsId}-metal)`} opacity=".9" />
      <path d="M78 78h72v83H78z" fill="#070d11" fillOpacity=".76" stroke="#78848b" />
      <path d="M90 131h48v13H90z" fill="#0a2433" stroke="#00aee8" />
      <path d="M96 136h31" stroke="#ff6a00" strokeWidth="2" />
      <Fan cx={155} cy={105} defsId={defsId} radius={31} />
      <path d="M82 86h26v27H82z" fill="#131a1e" stroke="#ff6a00" />
      <path d="M87 92h16m-16 6h16m-16 6h10" stroke="#69767e" />
      <circle cx="74" cy="165" r="8" fill="#050607" stroke="#ff6a00" strokeWidth="1.5" />
      <Fasteners points={[[68, 64], [162, 51], [68, 182], [162, 162]]} />
      </>}
    </TechnicalShell>
  );
}

function ProMax({ label, size }: Pick<Props, "label" | "size">) {
  return (
    <TechnicalShell label={label} model="pironman5-pro-max" size={size ?? 92}>
      {(defsId) => <>
      <path d="m32 35 148-13 28 22v151L32 202Z" fill={`url(#${defsId}-panel)`} stroke="#46535b" strokeWidth="2.2" />
      <path d="m180 22 28 22v151l-28-9Z" fill="#020405" stroke="#263239" />
      <path d="M47 49h55v78H47z" fill="#03151a" stroke="#00aee8" strokeWidth="1.5" />
      <path d="M54 58h40v6H54zm0 14h31v4H54zm0 11h36v4H54z" fill="#55dfed" opacity=".75" />
      <path d="M54 102h16l7 12H54z" fill="#ff6a00" opacity=".8" />
      <Oled x={51} y={136} />
      <path d="M105 57h45v14h-45zm0 22h45v14h-45z" fill="#0a2433" stroke="#00aee8" />
      <path d="M110 62h30m-30 22h30" stroke="#ff6a00" strokeWidth="2" />
      <Fan cx={169} cy={77} defsId={defsId} radius={22} />
      <Fan cx={169} cy={139} defsId={defsId} radius={22} />
      <g fill="#89959c">
        {Array.from({ length: 7 }, (_, index) => <circle key={index} cx={111 + index * 6} cy="168" r="1.4" />)}
      </g>
      <circle cx="46" cy="179" r="10" fill="#050607" stroke="#ff6a00" strokeWidth="1.5" />
      <path d="M36 197h151" stroke="#ff6a00" strokeWidth="2" />
      <Fasteners points={[[39, 42], [175, 29], [39, 195], [175, 180]]} />
      </>}
    </TechnicalShell>
  );
}

function Generic({ label, size }: Pick<Props, "label" | "size">) {
  return (
    <TechnicalShell label={label} model="generic" size={size ?? 92}>
      {() => <>
      <path d="M45 51h150v117H45z" fill="#080b0e" stroke="#72808a" strokeWidth="2" />
      <path d="M61 72h88m-88 27h88m-88 27h88" stroke="#66747d" opacity=".6" />
      <circle cx="174" cy="72" r="5" fill="#24d875" />
      <circle cx="174" cy="99" r="5" fill="#00aee8" />
      <circle cx="174" cy="126" r="5" fill="#ff6a00" />
      <path d="M85 184h70M99 168v16m42-16v16" stroke="#8c9aa3" strokeWidth="3" />
      </>}
    </TechnicalShell>
  );
}

export function PironmanDeviceIcon({ label, model, size = 92 }: Props) {
  switch (model) {
    case "pironman5":
      return <Standard label={label} size={size} />;
    case "pironman5-max":
      return <Max label={label} size={size} />;
    case "pironman5-mini":
      return <Mini label={label} size={size} />;
    case "pironman5-pro-max":
      return <ProMax label={label} size={size} />;
    default:
      return <Generic label={label} size={size} />;
  }
}
