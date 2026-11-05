All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] - 2020-04-30
### Added
- code.json metadata file
- DISCLAIMER.md
- CHANGELOG.md
- LICENSE
- tests
### Changed
- python 3 support only

## [2.2.5] - 2018-05-23
### Changed
- Adam J. Stewart suggestion/adds new CLI options for retries and directory control

## [2.2.4] - 2018-05-22
### Fixed
- default option for auto-retry

## [2.2.3] - 2018-05-21
### Added
- retry option for downloads

## [2.2.2] - 2018-05-21
### Changed
- include User-Agent header

## [2.2.1] - 2018-03-01
### Fixed
- bug in python3 requests usage

## [2.2.0] - 2018-01-31
### Changed
- Woodstonelee added option to download checksum and error handling on bad urls
- Updated HTTPS support for python 2.7 series (allow using requests library)

## [2.1.0] - 2016-09-18
### Changed
- Converted to using the ESPA API proper rather than relying on the RSS feed
- The downloads will now tell you which file number of all available scenes is being downloaded.
- Added a try/except clause for cases where the remote server closes the connection during a download.

## [2.0.1] - 2016-06-06
### Fixed
- fixed handling none arg

## [2.0.0] - 2016-06-03
### Added
- Guy Serbin added support for Python 3.x and download progress indicators.

## [1.1.0] - 2016-04-11
### Changed
- removed max retry verbage

## [1.0.0] - 2015-08-19
### Initial Release


