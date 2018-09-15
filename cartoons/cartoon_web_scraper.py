from bs4 import BeautifulSoup, SoupStrainer
import urllib.request as urllib
from skimage import io
import csv
from datetime import timedelta, date, datetime
import argparse


source_root = "https://www.gocomics.com/"
dest_root = "E:/ML/keras/data/cartoons/"

# inclusive
def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days + 1)):
        yield start_date + timedelta(n)


def scrape_data(name, datestring):
    source_path = source_root + name + '/' + datestring
    print('scraping from ', source_path)
    page = urllib.urlopen(source_path)
    soup = BeautifulSoup(page, 'html.parser')

    if soup.find("picture", class_="item-comic-image") is None:
        print('invalid id')
        return None, None
    
    # grab the image
    img_tag = soup.find("picture", class_="item-comic-image").img
    img_src = img_tag['src']
    img = io.imread(img_src)

    # grab the image size
    size = [len(img), len(img[0])]

    return img, [name, datestring.replace('/', '-'), size]


argparser = argparse.ArgumentParser()
argparser.add_argument('name', help='the name/url id of the cartoon')
argparser.add_argument('startdate', help='the inclusive start date, in the format YYYY-MM-DD')
argparser.add_argument('enddate', help='the inclusive end date, in the format YYYY-MM-DD')
args = argparser.parse_args()

start_date = datetime.strptime(args.startdate, '%Y-%m-%d')
end_date = datetime.strptime(args.enddate, '%Y-%m-%d')
name = args.name

with open(dest_root + 'metadata.csv', 'a', newline='') as file:
    writer = csv.writer(file, quotechar='|', quoting=csv.QUOTE_MINIMAL)
    i = 0 # TODO: remove this later, this is just to get about the same number of strips
    for date in daterange(start_date, end_date):
        # TODO: above for the below
        if (not (i%2 == 0)):
            img, metadata = scrape_data(name, date.strftime("%Y/%m/%d"))
            if img is not None:
                io.imsave(dest_root + 'images/' + name + date.strftime("%Y-%m-%d") + '.png', img)
                writer.writerow(metadata)
        i += 1
