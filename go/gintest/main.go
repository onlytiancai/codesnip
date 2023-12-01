package main

import (
	"fmt"

	"github.com/gin-gonic/gin"
  "gorm.io/gorm"
  "gorm.io/driver/sqlite"
)

type Product struct {
  gorm.Model
  Code  string
  Price uint
}

func MyHandler(c *gin.Context) {
  db, err := gorm.Open(sqlite.Open("test.db"), &gorm.Config{})
  if err != nil {
    panic("failed to connect database")
  }

  // 迁移 schema
  db.AutoMigrate(&Product{})
	fmt.Println("created product table")

  // Create
  db.Create(&Product{Code: "D42", Price: 100})
	fmt.Println("insert product")

  // Read
  var product Product
  db.First(&product, 1) // 根据整型主键查找
  fmt.Println("get by id:%s", product)
  db.First(&product, "code = ?", "D42") // 查找 code 字段值为 D42 的记录
  fmt.Println("get by code:%s", product)

  // Update - 将 product 的 price 更新为 200
  db.Model(&product).Update("Price", 200)
  fmt.Println("update product")

  // Update - 更新多个字段
  db.Model(&product).Updates(Product{Price: 200, Code: "F42"}) // 仅更新非零值字段
  fmt.Println("update product 2")
  db.Model(&product).Updates(map[string]interface{}{"Price": 200, "Code": "F42"})
  fmt.Println("update product 3")

  // Delete - 删除 product
  db.Delete(&product, 1)
  fmt.Println("delete product")

	c.JSON(200, gin.H{
		"hello": "Hi, nice to meet you.",
	})
}

func Login(c *gin.Context) {
	c.HTML(200, "login.html", nil)
}

func DoLogin(c *gin.Context) {
	username := c.PostForm("username")
	password := c.PostForm("password")

	c.HTML(200, "welcome.html", gin.H{
		"username": username,
		"password": password,
	})

}

type User struct {
	Username string   `form:"username"`
	Password string   `form:"password"`
	Hobby    []string `form:"hobby"`
	Gender   string   `form:"gender"`
	City     string   `form:"city"`
}

func Regsiter(c *gin.Context) {
	var user User
	c.ShouldBind(&user)
	c.String(200, "User:%s", user)
}

func GoRegister(c *gin.Context) {
	c.HTML(200, "register.html", nil)
}

func MyMiddleware1(c *gin.Context) {
	fmt.Println("我的第一个中间件")
}

func MyMiddleware2(c *gin.Context) {
	fmt.Println("我的第二个中间件")
}

func main() {
	e := gin.Default()
	e.LoadHTMLGlob("templates/*")
  e.Use(MyMiddleware1, MyMiddleware2)

  e.Static("/assets", "./assets")

	e.GET("/", MyHandler)
	e.GET("/login", Login)
	e.POST("/login", DoLogin)
	e.POST("/register", Regsiter)
	e.GET("/register", GoRegister)
	e.Run()
}
