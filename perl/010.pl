$n = 0;
while (defined($line = <>)) {
    $n += 1;
    chomp($line);
    print "$n $line\n";
}
