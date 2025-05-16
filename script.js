 // Smooth scrolling for navigation links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                document.querySelector(this.getAttribute('href')).scrollIntoView({
                    behavior: 'smooth'
                });
            });
        });
        
        // Initialize button functionality
        const initBtn = document.getElementById('initBtn');
        initBtn.addEventListener('click', () => {
            initBtn.innerHTML = '<i class="fas fa-cog fa-spin"></i> INITIALIZING...';
            initBtn.style.background = 'linear-gradient(45deg, var(--success), #2ecc71)';
            
            // Simulate initialization process
            setTimeout(() => {
                initBtn.innerHTML = '<i class="fas fa-check"></i> SYSTEM READY';
                initBtn.style.background = 'linear-gradient(45deg, var(--success), #2ecc71)';
                
                // After 3 seconds, reset the button
                setTimeout(() => {
                    initBtn.innerHTML = '<i class="fas fa-power-off"></i> INITIALIZE SYSTEM';
                    initBtn.style.background = 'linear-gradient(45deg, var(--danger), #b5179e)';
                }, 3000);
            }, 2000);
        });
        
        // Add scroll animation
        window.addEventListener('scroll', () => {
            const navbar = document.querySelector('.navbar');
            if (window.scrollY > 50) {
                navbar.style.background = 'rgba(26, 26, 46, 0.95)';
                navbar.style.boxShadow = '0 5px 20px rgba(0, 0, 0, 0.2)';
            } else {
                navbar.style.background = 'rgba(26, 26, 46, 0.9)';
                navbar.style.boxShadow = 'none';
            }
        });