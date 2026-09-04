import { useCallback, useEffect, useState } from "react";
import { apiRequest } from "../lib/api";
import type { AlertChannel } from "../components/agentTypes";

export interface AlertChannelTestResult {
  channel_id: string;
  delivered: boolean;
  error: string;
  detail: string;
}

/**
 * Data operations for alert-delivery channels.
 *
 * Kept apart from the automations hook: a channel is *where* an alert goes,
 * which is a different subject from the rule that fires it, and neither needs
 * the other to load. The component above owns the form fields; this owns the
 * server round-trips and the loaded list.
 */
export function useAlertChannels({
  csrfToken,
  enabled,
}: {
  csrfToken: string;
  /** Only administrators can reach these endpoints, so skip the load otherwise. */
  enabled: boolean;
}) {
  const [channels, setChannels] = useState<AlertChannel[]>([]);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");

  const load = useCallback(async () => {
    if (!enabled) return;
    try {
      const data = await apiRequest<{ channels: AlertChannel[] }>("/assistant/alert-channels");
      setChannels(Array.isArray(data?.channels) ? data.channels : []);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Delivery channels could not be loaded.");
    }
  }, [enabled]);

  useEffect(() => {
    void load();
  }, [load]);

  const createChannel = useCallback(async (body: Record<string, unknown>) => {
    setBusy(true); setNotice("");
    try {
      await apiRequest("/assistant/alert-channels", { method: "POST", body: JSON.stringify(body) }, csrfToken);
      setNotice("Delivery channel added. Send a test before you rely on it.");
      await load();
      return true;
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The delivery channel could not be added.");
      return false;
    } finally { setBusy(false); }
  }, [csrfToken, load]);

  const toggleChannel = useCallback(async (channel: AlertChannel) => {
    setBusy(true); setNotice("");
    try {
      await apiRequest(`/assistant/alert-channels/${channel.id}`, {
        method: "PATCH", body: JSON.stringify({ enabled: !channel.enabled }),
      }, csrfToken);
      await load();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The delivery channel could not be updated.");
    } finally { setBusy(false); }
  }, [csrfToken, load]);

  const deleteChannel = useCallback(async (channel: AlertChannel) => {
    setBusy(true); setNotice("");
    try {
      await apiRequest(`/assistant/alert-channels/${channel.id}`, { method: "DELETE" }, csrfToken);
      setNotice("Delivery channel removed.");
      await load();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The delivery channel could not be removed.");
    } finally { setBusy(false); }
  }, [csrfToken, load]);

  const testChannel = useCallback(async (channel: AlertChannel) => {
    setBusy(true); setNotice("");
    try {
      const result = await apiRequest<AlertChannelTestResult>(
        `/assistant/alert-channels/${channel.id}/test`, { method: "POST" }, csrfToken,
      );
      setNotice(result.delivered
        ? `Test alert delivered to ${channel.name}.`
        : `Test failed for ${channel.name}: ${result.error}`);
      await load();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "The test alert could not be sent.");
    } finally { setBusy(false); }
  }, [csrfToken, load]);

  return { channels, busy, notice, load, createChannel, toggleChannel, deleteChannel, testChannel };
}
