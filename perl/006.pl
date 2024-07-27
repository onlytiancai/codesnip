@array = qw# dino fred barney #;
print @array, "\n";
$m = shift (@array); #$m 得到 "dino"， @array 现在为 ("fred", "barney")
print @array, $m, "\n";
$n = shift @array; #$n 得到 "fred", @array 现在为 ("barney")
print @array, $m, $n, "\n";
shift @array; #@array 现在为空
$o = shift @array; #$o 得到 undef, @arry 仍为空
unshift(@array,5); #@array 现在为(5)
unshift @array,4; #@array 现在为(4,5)
@others = 1..3;
unshift @array, @others; #array 现在为(1,2,3,4,5)
print @array, @others, "\n";

@rocks = qw{ flintstone slate rubble };
print "quartz @rocks limestone\n"; #输出为 5 种 rocks 由空格分开

foreach $rock (qw/ bedrock slate lava /) {
    print "One rock is $rock.\n" ; #打印出 3 种 rocks
}

foreach(1..10){ #使用默认的变量$_
    print "I can count to $_!\n";
}

$_ = "Yabba dabba doo\n";
print; # 打印出默认变量 $_

@fred = 6 .. 10;
print "@fred\n";
@barney = reverse (@fred); #得到 10,9,8,7,6
print "@barney\n";
@wilma = reverse 6..10; #同上,没有使用额外的数组
print "@wilma\n";
@fred = reverse @fred; #将逆转过的字符串存回去
print "@fred\n";

@rocks = qw/ bedrock slate rubble granite /;
print "@rocks\n";
@sorted = sort(@rocks); #得到 bedrock, granite, rubble, slate
print "@sorted\n";

@people = qw( fred barney betty );
@sorted = sort @people; #列表内容： barney , betty, fred
$number = 42 + @people; #标量内容：42+3,得到 45
print "$number @sorted\n";

@backwards = reverse qw / yabba dabba doo /;
#返回 doo, dabba, yabba
print "@backwards\n";
$backwards = reverse qw/ yabba dabba doo /;
print "$backwards\n";
#返回 oodabbadabbay

@fred = 6*7;
print "@fred\n";
@barney = "hello" . '' . "world";
print "@barney\n";

@rocks = qw(talc quartz jade obsidian);
print "How many rocks do you have?\n";
print "I have  @rocks, rocks!\n"; # 错误,输出 rocks 的名字
print "I have ". scalar @rocks . " rocks!\n"; # 正确,输出其数字
