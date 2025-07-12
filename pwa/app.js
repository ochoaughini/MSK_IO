function checkLicense() {
    return localStorage.getItem('license') === 'valid';
}

function showScreen(screen) {
    if (!checkLicense()) {
        alert('Invalid license');
        return;
    }
    const content = document.getElementById('screen');
    content.innerHTML = '';
    if (screen === 'registration') {
        content.innerHTML = '<h2>Register Voice</h2><button onclick="recordVoice()">Record</button>';
    } else if (screen === 'send-sms') {
        content.innerHTML = '<h2>Send SMS</h2><input id="phone" placeholder="Phone"><button onclick="sendSMS()">Send</button>';
    } else if (screen === 'verify') {
        content.innerHTML = '<h2>Verify Token</h2><input id="token" placeholder="Token"><button onclick="verifyToken()">Verify</button>';
    }
}

async function recordVoice() {
    console.log('Recording voice...');
    const token = 'sample-token';
    localStorage.setItem('token', token);
}

async function sendSMS() {
    const phone = document.getElementById('phone').value;
    const token = localStorage.getItem('token');
    if (token) await sendSMSToken(phone, token);
}

async function verifyToken() {
    const token = document.getElementById('token').value;
    console.log(`Verifying ${token}`);
}
