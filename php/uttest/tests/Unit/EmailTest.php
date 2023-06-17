<?php

namespace Unit\Test;
use PHPUnit\Framework\TestCase;
use App\Email;

class EmailTest extends TestCase
{
    public function testCanBeCreatedFromEmailAddress(): void
    {
        $this->assertInstanceOf(
            Email::class,
            Email::fromString('user@example.com') 
        );
    }

    public function testCannotBeCreatedFromInvalidEmailAddress(): void
    {
        $this->expectException(\InvalidArgumentException::class);        
        Email::fromString('invalid');
    }

    public function testCanBeUsedAsstring(): void
    {
        $this->assertEquals(
            'user@example.com',
            Email::fromString('user@example.com')
        );    
    }
}
