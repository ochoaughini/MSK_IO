chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'sign') {
        console.log('Signing content...');
        sendResponse({ success: true });
    }
});
