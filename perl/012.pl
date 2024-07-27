$user = 'wawa';
$days_to_die = 100000;
printf "Hello, %s : your password expires in %d days!\n", $user, $days_to_die;
printf "%6d\n", 42; # 输出为 ○○○○ 42 (○此处指代空格)
printf "%.2f\n", 2e3+1.95; # 2001

my @items = qw( wilma dino pebbles );
my $format = "The items are:\n". ("%10s\n" x @items);
printf $format, @items;

printf "The items are:\n". ("%10s\n"x@items), @items;
print "items count:", scalar @items, "\n";
