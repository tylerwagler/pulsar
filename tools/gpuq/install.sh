#!/usr/bin/env bash
# install.sh -- install gpuq for this user: ~/.local/lib/gpuq/gpuq.py (a copy, so switching the checkout's
# branch never changes the running daemon), ~/bin/gpuq -> it, and the gpuq.service user unit (enabled,
# restarted; running jobs survive a restart and are re-adopted).
set -eu
src=$(cd "$(dirname "$0")" && pwd)
dest=$HOME/.local/lib/gpuq
mkdir -p "$dest" "$HOME/bin" "$HOME/.config/systemd/user"
install -m 755 "$src/gpuq.py" "$dest/gpuq.py.new"
mv -f "$dest/gpuq.py.new" "$dest/gpuq.py"
ln -sfn "$dest/gpuq.py" "$HOME/bin/gpuq"
install -m 644 "$src/gpuq.service" "$HOME/.config/systemd/user/gpuq.service"
systemctl --user daemon-reload
systemctl --user enable gpuq.service
systemctl --user restart gpuq.service
[ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null)" = yes ] ||
  echo "install.sh: Linger=no -- gpuq.service stops when $USER's last login session ends" \
       "(fix: sudo loginctl enable-linger $USER)" >&2
case ":$PATH:" in *":$HOME/bin:"*) ;; *) echo "install.sh: $HOME/bin is not on this PATH; use $HOME/bin/gpuq" >&2 ;; esac
