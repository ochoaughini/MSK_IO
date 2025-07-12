const puppeteer = require('puppeteer');
jest.setTimeout(30000);

describe('VoicePasskey Integration Tests', () => {
    let browser, page;

    beforeAll(async () => {
        browser = await puppeteer.launch();
        page = await browser.newPage();
    });

    afterAll(async () => await browser.close());

    test('SMS send/receive flow', async () => {
        await page.evaluate(() => sendSMSToken('1234567890', 'test-token'));
        const validated = await page.evaluate(() => parse_and_validate_sms('encoded-message-stub'));
        expect(validated).not.toBeNull();
    });

    test('Mobile screens navigation', async () => {
        await page.goto('http://localhost:3000/pwa/index.html');
        await page.click("button[onclick=\"showScreen('registration')\"]");
        const header = await page.$eval('#screen h2', el => el.textContent);
        expect(header).toBe('Register Voice');
    });

    test('Extension token import and verify', async () => {
        await page.goto('chrome-extension://extension-id/popup.html');
        await page.click('#import-token');
        await page.click('#verify');
    });

    test('Offline-first behavior', async () => {
        await page.setOfflineMode(true);
        await page.goto('http://localhost:3000/pwa/index.html');
        const app = await page.$('#app');
        expect(app).not.toBeNull();
    });
});
