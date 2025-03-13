import turtle

def draw_rose():
    window = turtle.Screen()
    window.bgcolor("white")

    pen = turtle.Turtle()
    pen.color("red")
    pen.speed(3)

    # 绘制玫瑰花瓣
    for _ in range(36):
        pen.circle(100, 60)
        pen.left(120)
        pen.circle(100, 60)
        pen.left(120)
        pen.right(10)

    # 绘制花茎
    pen.color("green")
    pen.right(90)
    pen.forward(300)

    pen.hideturtle()
    window.exitonclick()

draw_rose()