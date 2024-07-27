%some_hash = ("foo",35, "bar", 12.4, 2.5, "hello", "wilma", 1.72e30, "betty", "bye");
print "000 $some_hash{2.5}\n";
@array_array = %some_hash;
print "111 @array_array\n";

%new_hash = %some_hash;
@array_array = %new_hash;
print "222 @array_array\n";

%inverse_hash = reverse %new_hash;
@array_array = %inverse_hash;
print "333 @array_array\n";

my %last_name = (
    "fred" => "flintstone",
    "dino" => undef,
    "barney" => "rubble",
    "betty" => "rubble",
);

print "$last_name{'fred'}\n";

my %hash = ("a" => 1, "b" => 2, "c" => 3);
my @k = keys %hash;
my @v = values %hash;

print "@k | @v\n";

while (($key, $value) = each %hash) {
    print "$key => $value\n";
}

if (exists $hash{'a'}) {
    print "Hey, there's a libaray card for dino!\n";
}

delete $hash{'a'}; # 将$person 的借书卡删除掉
my @k = keys %hash;
print "000 @k \n";
