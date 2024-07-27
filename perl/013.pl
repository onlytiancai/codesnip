if (!open PASSWD, "/etc/passwd") {
    die "How did you get logged in?($!)";
}
while (<PASSWD>) {
    chomp;
    print "$_\n";
}
