import { z } from "zod";

const UserSchema = z.object({
  name: z.string().min(1),
  email: z.email(),
  age: z.number().int().min(18),
});

const user = UserSchema.parse({
  name: "Tom",
  email: "tom@example.com",
  age: 20,
});

console.log(user);
