$_ = "yabba dabba doo";
if (/abba/) {
    print "It matched!\n";
}

$_ = "I saw Barney\ndown at the bowing alley\nwith Fred\nlast night.\n";
if (/Barney.*Fred/s) {
    print "That string mentions Fred after Barney!\n";
}

my $some_other = "I dream of betty rubble.";
if ($some_other =~ /\brub/) {
    print "Aye, there's the rub.\n";
}

$_ = "Hello there, neighbor";
if (/\s(\w+),/) { #空格和逗号之间的词
    print "The word was $1 $` $'\n";
}

$_ = "He's out bowling with barney tonight.";
s/Barney/Fred/i; # Barney 被 Fred 替换掉
print "$_\n";
#接上例:现在 $_ 为 "He's out bowling with Fred tonight."
s/Wilma/Betty/; # 用 Wilma 替换 Betty(失败)
s/with (\w+)/agaist $1's team/;
print "$_\n"; # 为 "He's out bowling against Fred's team tonight.";

$_ = "home, sweet home!";
s/home/cave/g;
print "$_\n"; # "cave, sweet cave!"

$_ = "Input  data\t\tmay have extra whitespace.";
s/\s+/ /g; # 现在是 "Input data may have extra whitespace."
print "$_\n";

$file_name='../abc/';
print "000 $file_name\n";
$file_name =~ s#^\.*##s; # 将$file_name 中所有的 Unix 类型的路径去掉
print "111 $file_name\n";

$_ = "I saw Barney with Fred.";
s/(fred|barney)/\U$1/gi; # $_ 现在是 "I saw BARNEY with FRED."
print "$_\n";

$_ = "I saw Barney with Fred.";
s/(fred|barney)/\u\L$1/ig; #$_现在为 "I saw Fred with Barney."
print "$_\n";

$name='mArk';
print "Hello, \L\u$name\E, would you like to play a game?\n";

@fields = split /:/, "abc:def:g:h"; # 返回 ("abc", "def", "g", "h")
print "@fields\n";

my $some_input = "This is a \t test.\n";
my @args = split /\s+/, $some_input; # ("This", "is", "a", "test." )
print "@args|$args[0]\n";

my $x = join ":", 4, 6, 8, 10, 12; #$x 为 "4:6:8:10:12"
print "$x\n";

$_ = "Hello there, neighbor!";
my($first, $second, $third) =/(\S+) (\S+), (\S+)/;
print "$second is my $third\n" ;

my $text = "Fred dropped a 5 ton granite block on Mr. Slate";
my @words = ($text =~ /([a-z]+)/ig);
print "Result: @words\n";
#Result: Fred dropped a ton granite block on Mr slate

%names = ("Barney Rubble Fred Flintstone Wilma Flintstone" =~ /(\w+)\s(\w+)/g);
while (($k,$v) = each %names) {
    print "$k => $v\n";
}
