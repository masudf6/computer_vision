import cv2
import numpy as np
import sys

def task_one(color_space, image):
    color_space_map = {
        'XYZ': cv2.COLOR_BGR2XYZ,
        'HSV': cv2.COLOR_BGR2HSV,
        'Lab': cv2.COLOR_BGR2Lab,
        'YCrCb': cv2.COLOR_BGR2YCrCb,
        'gray': cv2.COLOR_BGR2GRAY
    }
    converted_image = cv2.cvtColor(image, color_space_map[color_space])
    channel1, channel2, channel3 = cv2.split(converted_image)
    gray = cv2.cvtColor(image, color_space_map['gray'])

    return gray, channel1, channel2, channel3

def task_two(scenic_img, green_img):
    # convert to hsv to target the green color
    hsv = cv2.cvtColor(green_img, cv2.COLOR_BGR2HSV)

    # set range for green color
    lower_green = np.array([45, 72, 52])
    upper_green = np.array([101, 255, 255])
    # mask the image: set everything out of range to 0
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # invert the mask to target the out of range part
    person = cv2.bitwise_not(mask) # inverse maask
    extracted_person = cv2.bitwise_and(green_img, green_img, mask=person)

    # white background
    white_background = np.full_like(green_img, 255)
    person_wb = cv2.add(extracted_person, cv2.bitwise_and(white_background, white_background, mask=mask))

    # scenic backgroud
    scenic_resized = cv2.resize(scenic_img, (green_img.shape[1], green_img.shape[0]))
    scenic_combined = cv2.bitwise_and(scenic_resized, scenic_resized, mask=mask)
    final_result = cv2.add(extracted_person, scenic_combined)

    return green_img, person_wb, scenic_resized, final_result

def resize_image(image):
    height, width = image.shape[:2]
    aspect_ratio = float(width/height)
    new_width = int(1290)    # width 1290 which is within the specification
    new_height = int(new_width / aspect_ratio)
    resized_image = cv2.resize(image, (new_width, new_height)) # default interpolation cv2.INTER_LINEAR

    return resized_image

def show_image(args):
    # extract Left Top, Right Top, Left Bottom, Right Bottom image from args
    LT, RT, LB, RB = args
    row_1 = np.concatenate((LT, RT), axis=1)
    row_2 = np.concatenate((LB, RB), axis=1)
    full = np.concatenate((row_1, row_2), axis=0)

    # resize the full image
    full_resized = resize_image(full)
    print(full_resized.shape)

    cv2.imshow(sys.argv[1][1:], full_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def validate_arguments(cmd_arguments):
    if len(cmd_arguments) != 3:
        print(f"Argument provided {len(cmd_arguments)}, 3 agruments excepted")
        sys.exit(1)

def main(cmd_arguments):
    color_spaces = ['XYZ', 'HSV', 'Lab', 'YCrCb']

    if cmd_arguments[1].lstrip('-') in color_spaces:
        color_space = cmd_arguments[1].lstrip('-')
        image = cv2.imread(cmd_arguments[2])
        show_image(task_one(color_space, image))
    else:
        scenic_img = cv2.imread(cmd_arguments[1])
        green_img = cv2.imread(cmd_arguments[2])
        show_image(task_two(scenic_img, green_img))


def parse_and_run():

    cmd_arguments = sys.argv

    validate_arguments(cmd_arguments)
    main(cmd_arguments)   

if __name__ == '__main__':
    parse_and_run()