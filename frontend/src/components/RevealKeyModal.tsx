import { useCallback, useRef, useState } from "react";
import { CodeBlock, IconTile } from "./ClusterPrimitives";
import { ClusterDialog } from "./ClusterDialog";
import { Icon } from "./Icon";
import { Button } from "./ui";
import "../styles/cluster-deployments.css";

/**
 * The one-time key reveal (F3c), drawn as the boards' "Shown once" dialog
 * (VD-200, ClusterDialogsManage).
 *
 * This is the ONLY place a plaintext API key ever appears in the UI. It is
 * handed the key as a prop and holds no copy of its own; when it closes, the
 * host drops the prop, so the string leaves React state with the unmounted
 * node. The reveal is the focal point while open — the copy button takes
 * initial focus so the value can be saved in one keystroke, and the warning is
 * the described-by so a screen reader hears "shown once" with the value.
 *
 * It is the cluster dialog (ModalShell beneath): body scroll locks, focus is
 * trapped, Escape and a backdrop press close it, and focus returns to the
 * trigger — all from `useDialogFocus`.
 */
export function RevealKeyModal({
  title,
  description,
  keyValue,
  onClose,
}: {
  title: string;
  description: string;
  keyValue: string;
  onClose: () => void;
}) {
  const copyRef = useRef<HTMLButtonElement>(null);
  const [copied, setCopied] = useState(false);

  const copy = useCallback(() => {
    if (!keyValue || !navigator.clipboard) return;
    void navigator.clipboard.writeText(keyValue).then(
      () => setCopied(true),
      () => setCopied(false),
    );
  }, [keyValue]);

  return (
    <ClusterDialog
      describedBy="reveal-key-warning"
      eyebrow="Shown once"
      footer={<Button onClick={onClose} type="button" variant="primary">Done</Button>}
      initialFocusRef={copyRef}
      onClose={onClose}
      title={title}
      titleId="reveal-key-title"
      tone="warning"
    >
      <div className="cl-reveal-icon"><IconTile accent name="key" /></div>
      <p>{description}</p>
      <div className="cl-reveal-value">
        <CodeBlock>{keyValue}</CodeBlock>
        <Button onClick={copy} ref={copyRef} type="button" variant="secondary">
          {copied ? "Copied" : "Copy key"}
        </Button>
      </div>
      <p className="cl-reveal-warning" id="reveal-key-warning">
        <Icon name="alert" size={16} />
        <span>This key is shown once. Store it now — Vaelor keeps only a fingerprint and cannot show it again.</span>
      </p>
    </ClusterDialog>
  );
}
