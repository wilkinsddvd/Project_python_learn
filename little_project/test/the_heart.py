import turtle

def draw_heart():
    window = turtle.Screen()
    window.bgcolor("white")

    pen = turtle.Turtle()
    pen.color("red")
    pen.speed(3)

    pen.begin_fill()
    pen.left(140)
    pen.forward(224)
    pen.circle(-112, 200)
    pen.left(120)
    pen.circle(-112, 200)
    pen.forward(224)
    pen.end_fill()

    pen.hideturtle()
    window.exitonclick()

draw_heart()