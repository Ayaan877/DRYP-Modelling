# espa-bulk-downloader

Automatically downloads all completed espa scenes.  Each scene is downloaded to the `--target_directory` and organized by order.

---

#### Support Information

This project is unsupported software provided by the U.S. Geological Survey (USGS) Earth Resources Observation and Science (EROS) Land Satellite Data Systems (LSDS) Project. For questions regarding products produced by this source code, please contact us at [custserv@usgs.gov][2].

#### Disclaimer

This software is preliminary or provisional and is subject to revision. It is being provided to meet the need for timely best science. The software has not received final approval by the U.S. Geological Survey (USGS). No warranty, expressed or implied, is made by the USGS or the U.S. Government as to the functionality of the software and related material nor shall the fact of release constitute any such warranty. The software is provided on the condition that neither the USGS nor the U.S. Government shall be held liable for any damages resulting from the authorized or unauthorized use of the software.


# Installation

## Requirements

* Python Version >= 3.6.4
* the Requests library: https://pypi.org/project/requests/


## Install
* Install with pip automatically:
```
pip install git+https://code.usgs.gov/espa/bulk-downloader.git
download_espa_order.py -h
```
* Clone this repository:
```
git clone https://code.usgs.gov/espa/bulk-downloader.git bulk-downloader
cd bulk-downloader
python ./download_espa_order.py -h
```
* Alternatively you can [download the stand alone zip][1] file which only requires python to run (see [requirements](#requirements))

# Usage
### Runtime options

Argument | Description
---|---
`-e or --email` | The email address used to submit the order
`-o or --order` | The order you wish to download.  (`ALL` or ESPA order-id)
`-d or --target_directory` | The local directory to store downloaded scenes
`-u or --username` | Your ERS username
`-p or --password` | Your ERS password
`-c or --checksum` | Download checksum files
`-r or --retry` | Retry instead of skipping failed files
`-n or --no-order-directories` | Store all files in one directory
`-v or --verbose` | Be vocal about process
`-i or --host` | If host downloading from is not https://espa.cr.usgs.gov 

> Linux/Mac Example: `python ./download_espa_order.py -d /some/directory/with/free/space -u your_username`

> Windows Example: `C:\python36\python download_espa_order.py -d C:\some\directory\with\free\space -u your_username`

# Notes
Retrieves all completed scenes for the user/order
and places them into the target directory.
Scenes are organized by order.

It is safe to cancel and restart the client, as it will
only download scenes one time (per directory)

If you intend to automate execution of this script,
please take care to ensure only 1 instance runs at a time.
Also please do not schedule execution more frequently than
once per hour.

# Running espa_bulk_downloader tests
python -m unittest test/test_download_espa_order.py

# References:
* IPDS Identifier: IP-117063 
* [ESPA Order Bulk Downloader Digital Object Identifier 10.5066/P9YBPYAR](https://doi.org/10.5066/P9YBPYAR)


[1]: https://code.usgs.gov/espa/bulk-downloader/-/archive/master/bulk-downloader-master.zip
[2]: mailto:custserv@usgs.gov
[3]: https://github.com/requests/requests
