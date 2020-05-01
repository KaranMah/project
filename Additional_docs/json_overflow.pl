use strict;
use warnings;
use File::Slurp;
use Jq;

my $file = $ARGV[0];
print $file ."\n";
my $file_size = -s $file;
print $file_size ."\n";
my @file_content = read_file($file);
my @data = jq '.', @file_content | jq -s ".[] | select((.datetime | contains(\"2014\")) and .is_retweet==false and .is_reply==false) | {query: .query, username: .usernameTweet, id: .ID, text: .text, nbr_retweet: .nbr_retweet, nbr_reply: .nbr_reply, nbr_favorite: .nbr_favorite, datetime: .datetime}" | jq -s;
open(my $fh, '>', 'temp_res.json');
print $fh @data;
close $fh;
print "Done...\n";
