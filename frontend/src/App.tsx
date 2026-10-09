import { useEffect, useState } from "react";
import type { Session } from "./types";
import { ApiError, apiRequest, SESSION_CHANGED_EVENT, SESSION_EXPIRED_EVENT } from "./lib/api";
import { AuthScreen } from "./components/AuthScreen";
import { Overview } from "./components/Overview";
import { WorkspaceErrorBoundary } from "./components/WorkspaceErrorBoundary";

type AppState = "loading" | "bootstrap" | "login" | "authenticated";

export default function App() {
  const [state, setState] = useState<AppState>("loading");
  const [session, setSession] = useState<Session | null>(null);
  const [error, setError] = useState("");
  // The refusal's code and status, so the sign-in screen can tell "enter a
  // code" from "that code was wrong" and from the sign-in limit (VD-200).
  const [errorCode, setErrorCode] = useState(""), [errorStatus, setErrorStatus] = useState(0);
  const [busy, setBusy] = useState(false);
  const [totpRequired, setTotpRequired] = useState(false);
  // The route the app-wide error boundary is keyed on, so a crash on one page
  // is tried again when the owner goes to another (review round 1).
  const [route, setRoute] = useState(() => window.location.hash);

  useEffect(() => {
    const sessionExpired = () => {
      setSession(null);
      setBusy(false);
      setTotpRequired(false);
      setError("Your session expired. Sign in again to continue.");
      setState("login");
    };
    window.addEventListener(SESSION_EXPIRED_EVENT, sessionExpired);
    // CR2: an administrator who lowered their own access carries on with the
    // re-read session; the shell remounts for the new role (see key below).
    const sessionChanged = (event: Event) => {
      const next = (event as CustomEvent<Session>).detail;
      if (next?.user) setSession(next);
    };
    window.addEventListener(SESSION_CHANGED_EVENT, sessionChanged);
    const routeChanged = () => setRoute(window.location.hash);
    window.addEventListener("hashchange", routeChanged);

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
    return () => {
      window.removeEventListener(SESSION_EXPIRED_EVENT, sessionExpired);
      window.removeEventListener(SESSION_CHANGED_EVENT, sessionChanged);
      window.removeEventListener("hashchange", routeChanged);
    };
  }, []);

  const submitAuth = async (username: string, password: string, totpCode: string) => {
    setBusy(true);
    setError("");
    setErrorCode("");
    setErrorStatus(0);
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
      if (authError instanceof ApiError) {
        if (authError.code === "totp_required") setTotpRequired(true);
        setErrorCode(authError.code);
        setErrorStatus(authError.status);
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

  /**
   * Sign out from the app-wide error screen (LESSONS 22). The shell may be the
   * thing that failed, so this never depends on it: when the control plane
   * cannot be reached the browser still leaves the session and says so.
   */
  const signOutFromError = async () => {
    try {
      await logout();
    } catch {
      setSession(null);
      setError("Sign-out did not reach the control plane, so the session may still be active there. Sign in again to continue.");
      setState("login");
    }
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
    // UX-A13: Home and the shell sit outside the workspace boundaries, so a
    // render error there left a blank page. This one catches the whole app.
    return (
      <WorkspaceErrorBoundary onSignOut={() => void signOutFromError()} workspaceKey={route}>
        <Overview key={session.user.role} onLogout={logout} session={session} />
      </WorkspaceErrorBoundary>
    );
  }

  return (
    <AuthScreen
      busy={busy}
      error={error}
      errorCode={errorCode}
      errorStatus={errorStatus}
      mode={state === "bootstrap" ? "bootstrap" : "login"}
      totpRequired={totpRequired}
      onSubmit={submitAuth}
    />
  );
}
