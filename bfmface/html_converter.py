import os
import base64
import shutil


def dump_html(img_paths, html_name, reversed=True, move_to_home=True):

    top = """
    <html>
    <head>
    <style>
    body
    {
        background-color:#d0e4fe;
        }
        h1
        {
            color:orange;
                text-align:center;
                }
                p
                {
                            font-family:"Times New Roman";
                                font-size:40px;
                                }
                </style>
    </head>
    <body>
    """
    bottom = """
    </body>
    </html>
    """

    print(img_paths)
    with open(html_name, "w") as f:
        f.write(top)
        img_paths = sorted(img_paths)
        if reversed:
            img_paths = img_paths[::-1]
        for path in img_paths:
            f.write("<p>")
            f.write(path)
            f.write("<br>")
            data_uri = base64.b64encode(open(path, 'rb').read()).decode('utf-8')
            img_tag = '<img src="data:image/png;base64,{0}">'.format(data_uri)
            f.write(img_tag)
            f.write("<br>")
            f.write("</p>\n")
        f.write(bottom)

    if move_to_home:
        # Move to UNC public html website.
        home_dir = os.path.expanduser("~")
        public_html_dir = os.path.join(home_dir, "public_html")
        if os.path.exists(public_html_dir):
            shutil.copy(
                html_name,
                public_html_dir,
            )



if __name__ == "__main__":
    for snap_name in os.listdir("snap"):
        dump_html(f"snap/{snap_name}")