"""Routes for each worker's software against its profile (VD-194 P1).

Split out of ``api_cluster_routes`` (1,000-line ceiling). Two routes, with the
fleet routes' roles:

* ``GET /cluster/worker-software`` (operator, like ``GET /cluster``): every
  enrolled worker's stored reading, projected into the card's words. A read of
  the controller's own store; it reaches no machine.
* ``POST /cluster/nodes/<id>/worker-software/recheck`` (administrator, like
  the existing Recheck): run the read-only profile probe on that worker now and
  store what it read. It runs a root program on the worker over SSH, which is
  why it is administrator-only; it writes nothing there itself, and when a
  part the profile owns differs it queues the update job (VD-194 P2).
"""

from __future__ import annotations

import logging
import sqlite3

from .api_common import ApiContext, payload
from .cluster_node_removal import failure_words
from .cluster_store import NODE_NOT_FOUND
from .ssh_transport import machine_failures
from .version import __version__
from .worker_profile import release_profile_digest, short
from .worker_profile_state import NOT_CHECKED_INTERNAL


LOGGER = logging.getLogger(__name__)

#: What a route says when the store or the machine could not be reached. Fixed
#: words: the cause, which can carry a database path, is logged (round 4 N2/N6).
_OUTAGES = (OSError, sqlite3.Error)


#: What the hardware Recheck answers when this controller itself failed. Fixed
#: words; the cause, which can quote a stored row, is logged (LESSONS 24).
RECHECK_INTERNAL = "Vaelor could not recheck this machine (an internal error, logged)."


def recheck_refusal(node_id: str, error: BaseException):
    """The hardware Recheck's answer to a failure (PH-R6), by kind.

    A machine that is gone is a 404; the machine, its sign-in or the link to it
    failing is a 400 in `failure_words`' words (paramiko's errors included);
    anything else - a node row that will not decode, a bug - is this
    controller's own fault: logged with its traceback, a 500 in fixed words.
    """
    if isinstance(error, ValueError) and str(error) == NODE_NOT_FOUND:
        return payload(error={"code": "cluster_refresh_failed", "message": NODE_NOT_FOUND}, status=404)
    if isinstance(error, machine_failures()):
        return payload(error={"code": "cluster_refresh_failed", "message": failure_words(error)}, status=400)
    if isinstance(error, (OSError, sqlite3.Error)):
        LOGGER.exception("the Recheck of node %s could not read its records", node_id)
    else:
        LOGGER.error("the Recheck of node %s stopped on an internal error", node_id,
                     exc_info=(type(error), error, error.__traceback__))
    return payload(error={"code": "cluster_refresh_failed", "message": RECHECK_INTERNAL}, status=500)


def register_worker_profile_routes(context: ApiContext) -> None:
    blueprint = context.blueprint
    callbacks = context.callbacks
    require_auth = context.require_auth

    def _unavailable():
        return payload(
            error={"code": "cluster_unavailable",
                   "message": "Worker software cannot be read: fleet management is not running."},
            status=503,
        )

    @blueprint.get("/cluster/worker-software")
    @require_auth("operator")
    def cluster_worker_software():
        manager = callbacks.get("cluster_manager")
        if manager is None:
            return _unavailable()
        # Round 2 B4 / round 4 N2: a release digest that cannot be worked out is
        # said beside the readings, and a store outage is a logged 503. A node
        # row whose JSON will not decode reads unknown for that worker alone
        # (P1-R4-1, `worker_software_all`). Any other failure is a logged 500
        # (LESSONS 10: this comment once promised "never a 500").
        try:
            release = release_profile_digest()
            release_line = "{} · profile {}".format(__version__, short(release))
        except OSError:
            LOGGER.exception("could not work out this release's worker profile")
            release = ""
            release_line = "{} · this release's worker profile could not be worked out (logged)".format(
                __version__)
        try:
            workers = manager.worker_software_all()
        except _OUTAGES:
            # Round 4 N2: an outage is a logged 503; anything else is a bug and
            # falls through to Flask's own logged 500.
            LOGGER.exception("could not list the workers' software readings")
            return payload(error={"code": "worker_software_unreadable",
                                  "message": "The workers' software readings could not be read right "
                                             "now (logged)."}, status=503)
        return payload({"controller_version": __version__, "release_profile": short(release),
                        "release_line": release_line, "workers": workers})

    @blueprint.post("/cluster/nodes/<node_id>/worker-software/recheck")
    @require_auth("administrator", csrf=True)
    def cluster_worker_software_recheck(node_id):
        manager = callbacks.get("cluster_manager")
        if manager is None:
            return _unavailable()
        try:
            return payload(manager.recheck_worker_profile(node_id))
        except ValueError as error:
            if str(error) == NODE_NOT_FOUND:
                return payload(error={"code": "worker_software_failed", "message": NODE_NOT_FOUND},
                               status=404)
            # P1-R4-2: recheck_worker_profile raises no other ValueError of its
            # own, so this is a node row that will not decode, or a bug - this
            # controller's fault, not the request's. Fixed words; the raw text,
            # which can quote the row, goes to the log (LESSONS 24).
            LOGGER.exception("the profile Recheck of node %s stopped on an internal error", node_id)
            return payload(error={"code": "worker_software_failed", "message": NOT_CHECKED_INTERNAL},
                           status=500)
        except _OUTAGES:
            # Round 3 R3-1 / round 4 N2: an outage is a logged 503 in fixed words;
            # a programming error falls through to Flask's own logged 500.
            LOGGER.exception("the profile Recheck of node %s could not run", node_id)
            return payload(error={"code": "worker_software_unreadable",
                                  "message": "This machine's software could not be checked right now "
                                             "(logged)."}, status=503)
