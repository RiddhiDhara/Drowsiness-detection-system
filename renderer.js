const { exec } = require('child_process');

function startDetection() {
    exec('python ./python/1.py', (err, stdout, stderr) => {
        if (err) {
            console.error(`Error: ${stderr}`);
            return;
        }
        console.log(`Output: ${stdout}`);
    });
}

