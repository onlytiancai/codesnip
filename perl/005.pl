$fred[0] = "yabba";
$fred[1] = "dabba";
$fred[2] = "doo";
print @fred, "\n";

@arr = qw ! fred barney betty wilma dino !;
print @arr, "\n";
@arr = qw ( fred barney betty wilma dino );
print @arr, "\n";

($fred, $barney, $dino) = ("flintstone", "rubble", 'ppp');
print $fred, $barney, $dino, "\n";
($fred, $barney) = qw <flintstone rubble slate granite>; #两个值被忽略了
print $fred, $barney,"\n";
($rocks[0],$rocks[1],$rocks[2],$rocks[3]) = qw/talc mica feldspar quartz/;

@rocks = qw / bedrock slate lava /;
print 'rocks:',@rocks, "\n";
@tiny = (); #空表
print 'tiny:', @tiny, "\n";
@giant = 1..10; #包含 10 个元素的表
print 'giant:', @giant, "\n";
@stuff = (@giant, 'hello', @giant); #包含 21 个元素的表
print 'stuff:', @stuff, "\n";
@dino = "granite";
print 'dino:',@dino, "\n";
@quarry = (@rocks, "crushed rock", @tiny, $dino);
print 'quarry:',@quarry, "\n";

@array = 5..9;
$fred = pop(@array); #$fred 得到 9,@array 现在为(5,6,7,8)
$barney = pop @array; #$barney gets 8, @array 现在为(5,6,7)
pop @array; #@array 现在为(5,6)(7 被丢弃了)
print @array, "\n";

push(@array,0); #@array 现在为(5,6,0)
push @array,8; #@array 现在为(5,6,0,8)
push @array,1..10; #@array 现在多了 10 个元素

print @array, "\n";
