import { useState } from "react";
import { Button, Input } from "./ui";
import { ClusterDialog } from "./ClusterDialog";
import { isPrivateIpv4 } from "./fleetTypes";
import "../styles/cluster-setup.css";

/**
 * The head-controller setup dialog (VD-200, the ClusterDialogsChange board):
 * choose the fixed private LAN address workers will use to join, then review
 * the initialize plan. The address is seeded from the controller's own
 * candidate only when that candidate is a private IPv4 — a public name or
 * hostname is never prefilled, because workers join by a stable private
 * address. The placeholder names the kind of address, never a literal one
 * (LESSONS 21).
 */

interface Props {
  candidateAddress?: string;
  busy: boolean;
  onClose: () => void;
  onReview: (advertiseAddress: string) => void;
}

export function InitializeControllerModal({ candidateAddress, busy, onClose, onReview }: Props) {
  const [address, setAddress] = useState(
    () => (candidateAddress && isPrivateIpv4(candidateAddress) ? candidateAddress : ""),
  );
  const valid = isPrivateIpv4(address);

  return (
    <ClusterDialog
      eyebrow="Head controller setup"
      footer={(
        <>
          <Button onClick={onClose}>Cancel</Button>
          <Button
            disabledReason={valid ? undefined : "Enter this node's private IPv4 address first."}
            onClick={() => onReview(address)}
            variant="primary"
          >
            Review controller plan
          </Button>
        </>
      )}
      onClose={() => { if (!busy) onClose(); }}
      title="Choose the controller LAN address"
      titleId="initialize-controller-title"
    >
      <p>Workers use this fixed private IPv4 address to join the cluster. Confirm that it belongs to this Vaelor node.</p>
      <Input
        inputMode="decimal"
        label="Controller LAN address"
        onChange={(event) => setAddress(event.target.value.trim())}
        placeholder="A private IPv4 address"
        value={address}
      />
      <div className="cl-panel">
        <p>
          <strong className="cl-strong">Use a stable address.</strong>{" "}
          Reserve this address in your router or configure a static LAN address before enrolling workers.
        </p>
      </div>
    </ClusterDialog>
  );
}
