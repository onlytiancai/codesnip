<?php
namespace Unit\Test;
use PHPUnit\Framework\TestCase;

class FirstTest extends TestCase
{
    public function testTrueAssetsToTrue()
    {
        $condtion = true;
        $this->assertTrue($condtion);    
    }
}
