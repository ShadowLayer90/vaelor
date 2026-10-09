import { createContext, useContext } from "react";

/*
 * Where the create forms of the schedules, alert-rule and delivery-channel
 * panels live (VD-200 decision 5).
 *
 * The Assistant's Schedules and alerts view opens each form in a dialog from a
 * button; Cluster > Activity keeps them inline in their cards. The Routines tab
 * provides "dialog" around the panels it is handed, so the page that builds
 * them does not have to know; with no provider (Cluster) a panel stays inline.
 * A panel's own `createIn` prop still wins over this.
 */
export type CreateIn = "inline" | "dialog";

export const RoutinesCreateIn = createContext<CreateIn>("inline");

export function useCreateIn(explicit?: CreateIn): CreateIn {
  const provided = useContext(RoutinesCreateIn);
  return explicit ?? provided;
}
