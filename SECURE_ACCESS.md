# Secure local and remote access

The deployed control plane uses HTTPS on port `34001` and WSS for authenticated
noVNC on port `34002`. The noVNC backend listens only on loopback; a bounded,
TLS-only proxy is the public transport boundary.

## First connection

Open:

`https://<appliance-ip>:34001/v2/` (for example `https://192.0.2.10:34001/v2/`
or `https://mini-pc.local:34001/v2/`)

### The household certificate authority

Each install creates its own certificate authority, the household root. No
other Vaelor shares it, and its private key never leaves the appliance: it is
kept in `/etc/vaelor/authority`, encrypted to that machine, and readable only by
root. The root signs the console's certificate, which names every address and
host name the appliance answers on. Addresses on Docker's and libvirt's own
bridges (`docker0`, `docker_gwbridge`, `br-…`, `virbr…`) are left out: they
answer only from the appliance's own containers. Vaelor renews that certificate
on its own, before it expires and when the machine gains an address or name it
does not cover. Each renewal is announced in every open console page about 15
minutes beforehand. A browser that trusts the root does not notice the renewal.
A browser that only accepted the warning page shows the warning again, and an
open page says that this is why it lost its connection. The console and the
browser desktop pick up a renewed certificate on the next connection, with no
restart.

The root can only vouch for private addresses (`10.0.0.0/8`, `172.16.0.0/12`,
the third RFC 1918 home-network range, loopback, `fc00::/7`) and local names (`.local`, `.lan`,
`.home.arpa`, `.internal`, `localhost` and the machine's own name). Those limits
are written into the root itself as name constraints. A device whose TLS
software honours them, as current browsers do, rejects anything the root signs
for a public address or a public domain name, even if the root's key were
stolen. A machine name that is itself a real top-level domain (such as `dev` or
`app`) is never added to the root. Tailscale addresses
(`100.64.0.0/10`) and `ts.net` names are outside that range, so a browser
reaching the console through them still warns.

### Trust it on your own devices

Vaelor's own machines trust the root automatically: the installer adds it to the
controller's trust store, and each worker gets it when it joins. Your own
computers and phones are yours to change, so Vaelor does not do it for you:

1. At the end of the install, the installer prints the root's SHA-256 and SHA-1
   fingerprints and one command for each of Windows, macOS and Linux. Each
   command downloads the root from the console and installs it only if its
   fingerprint matches the printed one. If the fingerprint does not match, it
   installs nothing.
2. The console shows the same commands under **Settings › Connections › Trust
   this Vaelor**, with a QR code a phone can scan to download the root.
3. The root's public certificate can be downloaded without signing in, at
   `https://<appliance-ip>:34001/api/v2/security/trust/root.crt`. Nothing else
   is served without signing in. Compare its fingerprint with the one the
   installer or the console shows before you trust it. The fingerprint check is
   what makes the download safe.

**If your browser will not open the console,** you do not need the console to
get these steps. They are saved, with the fingerprints, in
`/opt/vaelor/tls/trust-this-vaelor.txt` on the appliance: the installer writes the
file, and the household authority rewrites it whenever it is missing or the root
or the console's certificate changes, so it also stays current after a console
update. Read it over SSH with `sudo cat /opt/vaelor/tls/trust-this-vaelor.txt`. To
print the steps again, run this on the appliance. It names the address the
console's certificate covers:

```bash
sudo /opt/vaelor/venv/bin/python -m vaelor.tls_trust_commands
```

If you have no terminal on the appliance, most browsers still let you continue
past the warning once (**Advanced**, then **Continue** or **Proceed**). Open
**Settings › Connections › Trust this Vaelor** and run a command from there.
Check the fingerprint against one you read on the appliance itself: the
installer's last screen or the saved file. A page reached past a warning cannot
prove its own fingerprint.

Firefox keeps its own list of trusted roots. Either turn on its setting for
roots the operating system trusts (`security.enterprise_roots.enabled`), or
import the root under **Settings → Privacy & Security → Certificates**.

### Existing installs

An appliance installed before the household root keeps serving its previous
certificate for a while. The authority first gives every worker the root
alongside the certificate it serves today, and keeps the new console certificate
waiting. It switches the console to the new certificate:

- as soon as every worker the controller reached recently, over SSH or through
  its telemetry, holds the root;
- straight away when the controller's record says it has no workers;
- after up to 24 hours when there is no record of the workers at all.

A worker that is online but does not hold the root holds up the switch. To
switch anyway, run
`sudo /opt/vaelor/venv/bin/python -m vaelor.tls_authority promote`; that worker
then stops trusting the console until it gets the root. The authority never
switches to a certificate that does not cover the address the workers connect
to. The switch needs no restart. Your browser asks once more after it, unless
the device already trusts the root.

### Uninstalling and reinstalling

A keep-data uninstall keeps the household root, so after a reinstall your
devices still trust the console. A data purge removes the root, and the next
install creates a new one that every device must trust again. To keep the same
root across a purge or onto a new controller, export it first and import it
during the install:

```text
sudo /opt/vaelor/venv/bin/python -m vaelor.tls_authority export --output household.vaelor-authority
sudo ./deploy/install-vaelor.sh --import-authority household.vaelor-authority
```

The export is encrypted with a passphrase you choose. The installer asks for it,
or reads `VAELOR_AUTHORITY_PASSPHRASE` (use `sudo -E` to pass it). Importing
over a different root that is already on the machine is refused unless you add
`--replace-authority`. Every device that trusted the old root must then trust
the imported one.

## Internet access

Do not port-forward `34001` or `34002` directly from a public router. Use a
trusted VPN such as Tailscale/WireGuard, or an identity-aware reverse proxy.
Keep the dashboard and noVNC gateway reachable only from the private network or
VPN.

## Recovery

Vaelor portable-state exports and scheduled backups do not contain the
household root's key, the console's private TLS key or decryptable provider
secrets. After restoring onto another host, import the root you exported (see
above) or trust the new one, and reconnect provider credentials. Legacy `/opt/pironman5` backup
paths are read only by the documented migration tooling.
