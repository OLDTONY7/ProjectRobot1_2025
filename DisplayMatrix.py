from microbit import *


numbers = {
    '1': Image("00900:00900:00900:00900:00900"),
    '2': Image("09990:00090:09990:09000:09990"),
    '3': Image("09990:00090:09990:00090:09990")
}


while True:
    for num in ['1', '2', '3']:
        display.show(numbers[num])
        sleep(1000)
    sleep(2000)
