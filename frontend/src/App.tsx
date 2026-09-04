import { useEffect, useState } from "react";
import type { Session } from "./types";
import { ApiError, apiRequest, SESSION_EXPIRED_EVENT } from "./lib/api";
import { AuthScreen } from "./components/AuthScreen";
import { Overview } from "./components/Overview";

type AppState = "loading" | "bootstrap" | "login" | "authenticated";

export default function App() {
  const [state, setState] = useState<AppState>("loading");
  const [session, setSession] = useState<Session | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [totpRequired, setTotpRequired] = useState(false);

  useEffect(() => {
    const sessionExpired = () => {
      setSession(null);
      setBusy(false);
      setTotpRequired(false);
      setError("Your session expired. Sign in again to continue.");
      setState("login");
    };
    window.addEventListener(SESSION_EXPIRED_EVENT, sessionExpired);

    const initialize = async () => {
      try {
        const status = await apiRequest<{ bootstrap_required: boolean }>("/auth/status");
        if (status.bootstrap_required) {
          setState("bootstrap");
          return;
        }
        try {
          const active = await apiRequest<Session>("/auth/session");
          setSession(active);
          setState("authenticated");
        } catch (sessionError) {
          if (sessionError instanceof ApiError && sessionError.status === 401) {
            setState("login");
            return;
          }
          throw sessionError;
        }
      } catch (initializationError) {
        setError(
          initializationError instanceof Error
            ? initializationError.message
            : "The control plane could not be reached.",
        );
        setState("login");
      }
    };
    void initialize();
    return () => window.removeEventListener(SESSION_EXPIRED_EVENT, sessionExpired);
  }, []);

  const submitAuth = async (username: string, password: string, totpCode: string) => {
    setBusy(true);
    setError("");
    try {
      if (state === "bootstrap") {
        await apiRequest("/auth/bootstrap", {
          method: "POST",
          body: JSON.stringify({ username, password }),
        });
      }
      const active = await apiRequest<Session>("/auth/login", {
        method: "POST",
        body: JSON.stringify({ username, password, totp_code: totpCode }),
      });
      setSession(active);
      setState("authenticated");
    } catch (authError) {
      if (authError instanceof ApiError && authError.code === "totp_required") {
        setTotpRequired(true);
      }
      setError(
        authError instanceof Error ? authError.message : "Authentication failed.",
      );
    } finally {
      setBusy(false);
    }
  };

  const logout = async () => {
    if (!session) return;
    await apiRequest(
      "/auth/logout",
      { method: "POST", body: JSON.stringify({}) },
      session.csrf_token,
    );
    setSession(null);
    setState("login");
  };

  if (state === "loading") {
    return (
      <main className="loading-screen" role="status" aria-live="polite">
        <div className="loading-mark" />
        <span>Starting secure control plane</span>
      </main>
    );
  }

  if (state === "authenticated" && session) {
    return <Overview onLogout={logout} session={session} />;
  }

  return (
    <AuthScreen
      busy={busy}
      error={error}
      mode={state === "bootstrap" ? "bootstrap" : "login"}
      totpRequired={totpRequired}
      onSubmit={submitAuth}
    />
  );
}
