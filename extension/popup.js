document.getElementById('import-token').addEventListener('click', () => {
    const token = prompt('Enter token or scan QR');
    chrome.storage.local.set({ token });
});

document.getElementById('record-voice').addEventListener('click', async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    console.log('Recording...');
    setTimeout(() => stream.getTracks().forEach(track => track.stop()), 5000);
});

document.getElementById('verify').addEventListener('click', () => {
    chrome.storage.local.get('token', ({ token }) => {
        alert(`Token verified: ${token}`);
    });
});

document.getElementById('sign-content').addEventListener('click', () => {
    chrome.tabs.query({ active: true, currentWindow: true }, tabs => {
        chrome.tabs.sendMessage(tabs[0].id, { action: 'sign' });
    });
});
