use strict; #迫使采用更严格的检测
my $n = 0;
sub marine {
    $n += 1; #全局变量$n
    print "Hello, sailor number $n!\n";
}
&marine();
&marine();
&marine();
&marine();

sub max{
    my ($num) = @_; # 列表 context, 同($sum) = @_;
    print "max1 $num\n";
    my $num = @_; # 标量 context,同$num = @_;
    print "max2 $num\n";
    if($_[0] > $_[1]) {
        $_[0];
    } else {
        $_[1];
    }
}
$n = &max(10,15);
print "max n is $n\n";

foreach (1..10){
    my($square) = $_*$_; #本循环中的私有变量
    print "$_ squared is $square.\n";
}
