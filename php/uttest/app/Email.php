<?php
declare(strict_types=1)

namespace App;

final class Email
{
    private $email;

    private function __construct(string $email)
    {
        $this->isValidEmail($email);
        $this->email = $email;
    }

    public static function fromString(string $email): self
    {
        return new self($email);
    }

    public function __toString(): string
    {
        return $this->email;    
    }

    private function isValidEmail(string $email):void
    {
        if (!filter_var($email, FILTER_VALIDATE_EMAIL)) {
            throw new \InvalidArgumentException(
                springf(
                    '"%s" is not valid email address',
                    $email
                )
            );
        } 
    }
}
