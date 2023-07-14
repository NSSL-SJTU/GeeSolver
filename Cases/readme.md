
# 1. Text Captcha Schemes Used by Famous Websites and IoT Devices

All screenshots below were collected on **October 27, 2022** from the official website. On these websites, after multiple consecutive incorrect passwords are entered, the user will be required to submit both the password and the results of text-based captchas.  We have emailed the security administrators of these eight websites about the security risks of their captcha scheme.

## 1.1 Google
Google offers reCAPTCHA v2 (e.g., select all squares with street signs) and v3 (i.e., risk analysis according to the critical steps of the user journey) to protect websites. However, due to the good user-friendliness of text captchas, Google still uses text-based captchas on the user login page. Password blasting attacks must rely on a captcha solver with a high accuracy rate.

url: https://accounts.google.com/

screenshot: 

<img src="https://github.com/Anonymous-GeeSolver/GeeSolver-CAPTCHA/blob/main/Cases/google_en.png" width="300px">

## 1.2 Yandex
url: https://passport.yandex.eu/auth/restore/password/captcha

screenshot: 

<img src="https://github.com/Anonymous-GeeSolver/GeeSolver-CAPTCHA/blob/main/Cases/Yandex.png" width="300px">

## 1.3 Microsoft
url: https://login.live.com/ppsecure/post.srf

screenshot: 

<img src="https://github.com/Anonymous-GeeSolver/GeeSolver-CAPTCHA/blob/main/Cases/microsoft_en.png" width="300px">

## 1.4 Wikipedia
url: https://en.wikipedia.org/w/index.php?title=Special:UserLogin&returnto=Main+Page

screenshot: 

<img src="https://github.com/Anonymous-GeeSolver/GeeSolver-CAPTCHA/blob/main/Cases/wikipedia_en.png" width="300px">

## 1.5 Weibo
url: https://weibo.com/login.php/

screenshot: 

<img src="https://github.com/Anonymous-GeeSolver/GeeSolver-CAPTCHA/blob/main/Cases/weibo.png" width="300px">

## 1.6 Sina
url: https://security.weibo.com/iforgot/loginname?entry=sso

screenshot: 

<img src="https://github.com/Anonymous-GeeSolver/GeeSolver-CAPTCHA/blob/main/Cases/sina.png" width="300px">

## 1.7 Apple
url: https://appleid.apple.com/account

screenshot: 

<img src="https://github.com/Anonymous-GeeSolver/GeeSolver-CAPTCHA/blob/main/Cases/apple_en.png" width="300px">

## 1.8 Ganji
url: https://passport.58.com/login/

screenshot: 

<img src="https://github.com/Anonymous-GeeSolver/GeeSolver-CAPTCHA/blob/main/Cases/58.png" width="300px">

## 1.9 ASUS (Router)
url: http://router.asus.com/ (Log in from the intranet)

screenshot: 

<img src="https://github.com/Anonymous-GeeSolver/GeeSolver-CAPTCHA/blob/main/Cases/asus.png" width="300px">

# 2. Security Features of Eight Target Captcha Schemes


|  Scheme   |                                                             Example                                                             | CCT | Two-layer | Overlapping | Rotation | Distortion | Hollow Font | Variable Length | Occluding Line | Noisy |
|:---------:|:-------------------------------------------------------------------------------------------------------------------------------:|:---:|:---------:|:-----------:|:--------:|:----------:|:-----------:|:---------------:|:--------------:|:-----:|
|  Google   |  <img src="https://github.com/Anonymous-GeeSolver/GeeSolver-CAPTCHA/blob/main/images/google.jpg" width="120px" height="40px">   | ✔  |    ✘     |     ✔      |    ✔    |     ✔     |     ✘      |       ✔        |       ✘       |  ✘   |
|  Yandex   |  <img src="https://github.com/Anonymous-GeeSolver/GeeSolver-CAPTCHA/blob/main/images/yandex.png" width="120px" height="40px">   | ✔  |    ✘     |     ✔      |    ✔    |     ✔     |     ✔      |       ✔        |       ✔       |  ✘   |
| Microsoft | <img src="https://github.com/Anonymous-GeeSolver/GeeSolver-CAPTCHA/blob/main/images/microsoft.jpg" width="120px" height="40px"> | ✘  |    ✔     |     ✔      |    ✔    |     ✔     |     ✔      |       ✔        |       ✘       |  ✔   |
| Wikipedia | <img src="https://github.com/Anonymous-GeeSolver/GeeSolver-CAPTCHA/blob/main/images/wikipedia.png" width="120px" height="40px"> | ✘  |    ✘     |     ✘      |    ✔    |     ✔     |     ✘      |       ✔        |       ✘       |  ✘   |
|   Weibo   |   <img src="https://github.com/Anonymous-GeeSolver/GeeSolver-CAPTCHA/blob/main/images/weibo.jpg" width="120px" height="40px">   | ✘  |    ✘     |     ✔      |    ✔    |     ✔     |     ✘      |       ✘        |       ✘       |  ✘   |
|   Sina    |   <img src="https://github.com/Anonymous-GeeSolver/GeeSolver-CAPTCHA/blob/main/images/sina.png" width="120px" height="40px">    | ✘  |    ✘     |     ✔      |    ✔    |     ✘     |     ✔      |       ✘        |       ✔       |  ✘   |
|   Apple   |   <img src="https://github.com/Anonymous-GeeSolver/GeeSolver-CAPTCHA/blob/main/images/apple.jpg" width="120px" height="40px">   | ✘  |    ✘     |     ✔      |    ✔    |     ✘     |     ✘      |       ✔        |       ✘       |  ✔   |
|   Ganji   |  <img src="https://github.com/Anonymous-GeeSolver/GeeSolver-CAPTCHA/blob/main/images/ganji-1.png" width="120px" height="40px">  | ✘  |    ✘     |     ✘      |    ✘    |     ✔     |     ✘      |       ✘        |       ✔       |  ✔   |
