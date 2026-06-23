const nodemailer = require('nodemailer');

module.exports = async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const {
    name,
    email,
    organisation,
    institution,
    track,
    enquiry_type,
    message,
  } = req.body || {};

  if (!name || !email || !message) {
    return res.status(400).json({ error: 'Missing required fields' });
  }

  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(email)) {
    return res.status(400).json({ error: 'Invalid email address' });
  }

  const lines = [
    `Name:           ${name}`,
    `Email:          ${email}`,
    organisation ? `Organisation:   ${organisation}` : null,
    institution  ? `Institution:    ${institution}`  : null,
    track        ? `Track:          ${track}`         : null,
    enquiry_type ? `Enquiry type:   ${enquiry_type}` : null,
    '',
    'Message:',
    message,
  ].filter(l => l !== null).join('\n');

  const transporter = nodemailer.createTransport({
    host:   process.env.SMTP_HOST,
    port:   parseInt(process.env.SMTP_PORT || '465', 10),
    secure: (process.env.SMTP_PORT || '465') === '465',
    auth: {
      user: process.env.SMTP_USER,
      pass: process.env.SMTP_PASS,
    },
  });

  await transporter.sendMail({
    from:    `"House of Ideas" <${process.env.SMTP_USER}>`,
    to:      'aashish@perfectskills.in',
    replyTo: `${name} <${email}>`,
    subject: `[HoI] ${enquiry_type || 'New enquiry'} — ${name}`,
    text:    lines,
  });

  return res.status(200).json({ ok: true });
};
