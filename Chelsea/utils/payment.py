html_doge_wallet = """
<!DOCTYPE html>
<html lang="en">
<head>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
            margin: 0;
            padding: 0;
            color: #333;
        }
        .container {
            max-width: 600px;
            margin: 20px auto;
            background: #ffffff;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        h1 {
            color: #ba9b37;
        }
        .doge-wallet {
            font-size: 18px;
            color: #555;
            background: #f9f9f9;
            padding: 10px;
            border: 1px dashed #ba9b37;
            border-radius: 5px;
            margin: 10px auto;
            display: inline-block;
            word-break: break-all;
        }
        .qr-code {
            margin: 20px 0;
        }
        .note {
            font-size: 14px;
            color: #777;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Support the Project</h1>
        <p>If you’d like to support Chelsea, consider donating Dogecoin! Your help keeps the project alive and thriving.</p>
        <h3>Dogecoin Wallet</h3>
        <div class="doge-wallet">DBsfDMdbJGcd2XS1gXBMYJkrXmMCpqKJ1K</div>
        <div class="qr-code">
            <img src="https://chart.googleapis.com/chart?chs=200x200&cht=qr&chl=DBsfDMdbJGcd2XS1gXBMYJkrXmMCpqKJ1K&choe=UTF-8" alt="Dogecoin Wallet QR Code">
        </div>
        <p class="note">Scan the QR code or copy the wallet address to send Dogecoin.</p>
    </div>
</body>
</html>
"""