sub valid() {
    $_[0] eq'ok';
}
sub error() {
    print "error: $_[0]\n";
}
$input = 'err';
&error ("Invalid input") unless &valid($input);

$i=1;
$j=10;
$i += 2 until $i > $j;
print "$i $j\n";

$n=0;
print "", ($n += 2) while $n <10;
print "\n";

@person = qw /a b c/;
print "$_ " foreach @person;
print "\n";

my @people = qw{ fred barney fred Wilma dino barney fred pebbles};
my %count; #新的空的 hash
$count{$_}++ foreach @people; #根据情况创建新的 keys 和 values

while(($k,$v)=each %count) {
    print "$k=>$v\n";
}

my $width =16;
my $size =
($width < 10 ) ? "small":
($width < 20) ? "medium":
($width < 50) ? "large":
"extra_large"; #default

print "size = $size\n";
