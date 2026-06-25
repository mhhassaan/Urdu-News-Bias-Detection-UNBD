import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Urdu News Bias Detection",
  description: "Detect bias in Urdu news",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}
