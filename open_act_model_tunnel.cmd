@echo off
echo Keep this window open during ACT tests. Enter the Ubuntu account password when asked.
ssh -o ExitOnForwardFailure=yes -o ServerAliveInterval=15 -o UserKnownHostsFile=E:/Dobot/.ssh_known_hosts_training -N -L 127.0.0.1:8787:127.0.0.1:8787 yxy@192.168.5.100
pause
