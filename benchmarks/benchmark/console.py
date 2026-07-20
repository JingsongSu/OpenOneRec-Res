from rich.console import Console
from pyfiglet import Figlet

console = Console()
err_style = "bold red"
warning_style = "bold yellow"
success_style = "green"
dim_style = "dim"

# benchmark dataset
f = Figlet(font='digital')
head_print = lambda x : f.renderText(x)

head_style = "bold white on blue"
subhead_style = "bold black on bright_blue"
row_style = "black on bright_white"


# Generator styles
head_style_2 = "bold white on magenta"
subhead_style_2 = "white"