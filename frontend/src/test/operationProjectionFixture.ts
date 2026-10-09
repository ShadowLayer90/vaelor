import rawFixture from "./fixtures/operation-projection-v1.json";
import { isOperationProjection, type OperationProjection } from "../lib/operationOwner";

const parsedFixture: unknown = rawFixture;
if (!isOperationProjection(parsedFixture)) {
  throw new Error("The shared vaelor.operation.v1 fixture does not satisfy the frontend wire contract.");
}

export const canonicalOperationFixture: OperationProjection = parsedFixture;

/*
 * The canonical JSON carries fixed timestamps because it is a wire-contract
 * fixture: its job is to pin the *shape* of `vaelor.operation.v1`, and its
 * dates must not drift. But a test rendering it means "an operation as it
 * exists right now", and the card now refuses to claim progress for an
 * operation that has reported nothing in an hour (`operationHasStalled`) —
 * so a fixture frozen in the past renders as abandoned and every progress
 * assertion fails.
 *
 * The factory therefore hands back recent timestamps by default, which is
 * what "an operation" means in a test that goes on to assert a progress bar.
 * A test about staleness overrides `timestamps` explicitly and gets exactly
 * what it asks for.
 */
function recentTimestamps(): OperationProjection["timestamps"] {
  const now = Date.now();
  return {
    ...canonicalOperationFixture.timestamps,
    created_at: new Date(now - 74_000).toISOString(),
    started_at: new Date(now - 72_000).toISOString(),
    updated_at: new Date(now - 60_000).toISOString(),
  };
}

export function makeOperationFixture(overrides: Partial<OperationProjection> = {}): OperationProjection {
  const actionEndpoints = { ...canonicalOperationFixture.action_endpoints, ...overrides.action_endpoints };
  const owner = { ...canonicalOperationFixture.owner, ...overrides.owner };
  const operation: OperationProjection = {
    ...canonicalOperationFixture,
    timestamps: overrides.timestamps ?? recentTimestamps(),
    ...overrides,
    canonical_state: overrides.state ?? overrides.canonical_state ?? canonicalOperationFixture.canonical_state,
    owner,
    owner_route: overrides.owner_route ?? owner.route,
    owner_resource: overrides.owner_resource !== undefined ? overrides.owner_resource : owner.resource,
    progress: { ...canonicalOperationFixture.progress, ...overrides.progress },
    permissions: { ...canonicalOperationFixture.permissions, ...overrides.permissions },
    action_endpoints: actionEndpoints,
    endpoints: overrides.endpoints
      ? { ...canonicalOperationFixture.endpoints, ...overrides.endpoints }
      : actionEndpoints,
  };
  // #180: the wire always carries `staleness`, but this factory omits it unless
  // a test asks for it - the same reason it hands back recent timestamps rather
  // than the fixture's frozen ones. A staleness test either sets the server
  // verdict explicitly (and `operationHasStalled` renders it) or leaves it off
  // and exercises the client fallback over `timestamps`.
  if (overrides.staleness === undefined) {
    delete operation.staleness;
  }
  return operation;
}
